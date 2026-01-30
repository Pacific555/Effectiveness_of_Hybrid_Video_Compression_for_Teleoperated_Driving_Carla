#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down

    L            : toggle next light type
    SHIFT + L    : toggle high beam
    Z/X          : toggle right/left blinker
    I            : toggle interior light

    TAB          : change camera position

    R            : toggle recording images to disk

    CTRL + R     : toggle recording of simulation (replacing any previous)
    CTRL + P     : start replaying last recorded simulation
    CTRL + +     : increments the start time of the replay by 1 second (+SHIFT = 10 seconds)
    CTRL + -     : decrements the start time of the replay by 1 second (+SHIFT = 10 seconds)

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla
from carla import ColorConverter as cc

import argparse
import os
import sys
import time
import collections
import datetime
import logging
import math
import weakref
from collections import deque
import cv2
import subprocess
import threading
import pandas as pd
import re
import io
import PyNvVideoCodec as nvc
import av


import select

os.environ['PATH'] = r"C:\ffmpeg\bin" + os.pathsep + os.environ['PATH']

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_TAB
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_w
    from pygame.locals import K_a
    from pygame.locals import K_s
    from pygame.locals import K_d
    from pygame.locals import K_q
    from pygame.locals import K_m
    from pygame.locals import K_COMMA
    from pygame.locals import K_PERIOD
    from pygame.locals import K_p
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_z
    from pygame.locals import K_x
    from pygame.locals import K_r
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================

class World(object):

    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self.restart(args)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self, args):

        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713

        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # Get the ego vehicle
        while self.player is None:
            print("Waiting for the ego vehicle...")
            time.sleep(1)
            possible_vehicles = self.world.get_actors().filter('vehicle.*')
            for vehicle in possible_vehicles:
                if vehicle.attributes['role_name'] == args.rolename:
                    print("Ego vehicle found")
                    self.player = vehicle
                    break

        self.player_name = self.player.type_id

        # Set up the sensors.
        #self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.collision_sensor = CollisionSensor(self.player, self.hud, self.gnss_sensor)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        self.world.wait_for_tick()

    def tick(self, clock, wait_for_repetitions):
        if len(self.world.get_actors().filter(self.player_name)) < 1:
            if not wait_for_repetitions:
                return False
            else:
                self.player = None
                self.destroy()
                self.restart()

        self.hud.tick(self, clock)
        return True

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        sensors = [
            self.camera_manager.sensor_rgb,
            self.camera_manager.sensor_seg,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()
        if hasattr(self, "video_writer"):
            self.video_writer.release()
        if hasattr(self.camera_manager, "close_encoder"):
            self.camera_manager.close_encoder()
        if hasattr(self.camera_manager, "encoder"):
            final_pkt = self.camera_manager.encoder.EndEncode()
            if final_pkt and len(final_pkt) > 0:
                self.camera_manager.out_file.write(final_pkt)
                self.camera_manager.out_file.close()

        self.camera_manager.close_encoder()

        


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        self._control = carla.VehicleControl()
        self._lights = carla.VehicleLightState.NONE
        self._steer_cache = 0.0
        world.player.set_autopilot(self._autopilot_enabled)
        world.player.set_light_state(self._lights)
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock):
        current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_q:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.key == K_m:
                    self._control.manual_gear_shift = not self._control.manual_gear_shift
                    self._control.gear = world.player.get_control().gear
                    world.hud.notification('%s Transmission' %
                                            ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                elif self._control.manual_gear_shift and event.key == K_COMMA:
                    self._control.gear = max(-1, self._control.gear - 1)
                elif self._control.manual_gear_shift and event.key == K_PERIOD:
                    self._control.gear = self._control.gear + 1
                elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                    self._autopilot_enabled = not self._autopilot_enabled
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification(
                        'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                    current_lights ^= carla.VehicleLightState.Special1
                elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                    current_lights ^= carla.VehicleLightState.HighBeam
                elif event.key == K_l:
                    # Use 'L' key to switch between lights:
                    # closed -> position -> low beam -> fog
                    if not self._lights & carla.VehicleLightState.Position:
                        world.hud.notification("Position lights")
                        current_lights |= carla.VehicleLightState.Position
                    else:
                        world.hud.notification("Low beam lights")
                        current_lights |= carla.VehicleLightState.LowBeam
                    if self._lights & carla.VehicleLightState.LowBeam:
                        world.hud.notification("Fog lights")
                        current_lights |= carla.VehicleLightState.Fog
                    if self._lights & carla.VehicleLightState.Fog:
                        world.hud.notification("Lights off")
                        current_lights ^= carla.VehicleLightState.Position
                        current_lights ^= carla.VehicleLightState.LowBeam
                        current_lights ^= carla.VehicleLightState.Fog
                elif event.key == K_i:
                    current_lights ^= carla.VehicleLightState.Interior
                elif event.key == K_z:
                    current_lights ^= carla.VehicleLightState.LeftBlinker
                elif event.key == K_x:
                    current_lights ^= carla.VehicleLightState.RightBlinker

        if not self._autopilot_enabled:
            self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
            self._control.reverse = self._control.gear < 0
            # Set automatic control-related vehicle lights
            if self._control.brake:
                current_lights |= carla.VehicleLightState.Brake
            else: # Remove the Brake flag
                current_lights &= ~carla.VehicleLightState.Brake
            if self._control.reverse:
                current_lights |= carla.VehicleLightState.Reverse
            else: # Remove the Reverse flag
                current_lights &= ~carla.VehicleLightState.Reverse
            if current_lights != self._lights: # Change the light state only if necessary
                self._lights = current_lights
                world.player.set_light_state(carla.VehicleLightState(self._lights))
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.1, 1.00)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        # type: (carla.World, pygame.time.Clock) -> None
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        self._info_text += [
            ('Throttle:', c.throttle, 0.0, 1.0),
            ('Steer:', c.steer, -1.0, 1.0),
            ('Brake:', c.brake, 0.0, 1.0),
            ('Reverse:', c.reverse),
            ('Hand brake:', c.hand_brake),
            ('Manual:', c.manual_gear_shift),
            'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================

'''
class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

'''
class CollisionSensor(object):
    def __init__(self, parent_actor, hud, gnss_sensor):  # ⬅️ נוספה קליטת GNSS
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        self.gnss = gnss_sensor  # ⬅️ שמירה לשימוש פנימי
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)

        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

        # ✅ הוספת שמירה לקובץ כולל מיקום גיאוגרפי
        lat = getattr(self.gnss, 'lat', 0.0)
        lon = getattr(self.gnss, 'lon', 0.0)
        with open("collisions.csv", "a") as f:
            f.write(f"{event.frame},{actor_type},{intensity:.2f},{lat:.6f},{lon:.6f}\n")

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None

        # If the spawn object is not a vehicle, we cannot use the Lane Invasion Sensor
        if parent_actor.type_id.startswith("vehicle."):
            self._parent = parent_actor
            self.hud = hud
            world = self._parent.get_world()
            bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
            self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid circular
            # reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================


class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z

        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=bound_x + 0.05, z=bound_z+0.05),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        # points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

class CameraManager(object):
    def __init__(self, parent_actor, hud):

        self.rgb_array = None
        self.seg_array = None
        self.semantic_id_array = None
        self.process_start_time = None
        self.avg_latency_ms = 0.0
        self.frame_times = deque(maxlen=100)

        # GPU encoder and decoder components
        self.gpu_id = 0
        self.width, self.height = hud.dim
        self.fps = 20.0
        self.dt = 1.0 / self.fps

        # Encoder setup
        config = {
            "codec": "hevc",
            "bitrate": 1000000,
            "fps": self.fps,
            "preset": "P4",
            "gop": 1,  # I-frames only for low latency
            "bframes": 0,
            "tuning_info": "low_latency"
        }
        self.encoder = nvc.CreateEncoder(
            gpuid=self.gpu_id,
            width=self.width,
            height=self.height,
            fmt="ARGB",
            usecpuinputbuffer=True,
            **config
        )

        # Header cache for decoding
        self.hdr_cache = bytearray()  # Store VPS+SPS+PPS+IDR
        self.got_header = False  # Track if header is cached

        # Feeder class for NVDEC
        class Feeder:
            def __init__(self):
                self.buf = bytearray()
                self.pos = 0

            def reset(self, first_bytes: bytes):
                self.buf = bytearray(first_bytes)
                self.pos = 0
                print(f"[FEED] Reset with {len(first_bytes)} bytes")

            def append(self, b: bytes):
                self.buf.extend(b)
                print(f"[FEED] Appended {len(b)} bytes, total buffer={len(self.buf)}")

            def feed(self, dst: bytearray) -> int:
                n = min(len(dst), len(self.buf) - self.pos)
                if n:
                    dst[:n] = self.buf[self.pos:self.pos + n]
                    self.pos += n
                    print(f"[FEED] Fed {n} bytes, pos={self.pos}, remaining={len(self.buf) - self.pos}")
                return n

            def consume(self, n: int):
                self.buf = self.buf[self.pos + n:]
                self.pos = 0
                print(f"[FEED] Consumed {n} bytes, new buffer size={len(self.buf)}")

            def has_data(self):
                return self.pos < len(self.buf)

            def clear(self):
                self.buf = bytearray()
                self.pos = 0
                print("[FEED] Buffer cleared")

        self.feeder = Feeder()
        self.demux = None
        self.decoder = None

        # Output for debugging
        self.output_path = "output.h265"
        self.out_file = open(self.output_path, "wb")

        # Video writer for raw hybrid video
        self.video_writer = None
        self.output_raw_path = "hybrid_raw.avi"

        # ----------------------------------------------------

        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 1.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-2.0 * bound_x, y=+0.0 * bound_y, z=2.0 * bound_z),
                             carla.Rotation(pitch=8.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=+0.8 * bound_x, y=+0.0 * bound_y, z=1.3 * bound_z)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=+1.9 * bound_x, y=+1.0 * bound_y, z=1.2 * bound_z)),
             Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-2.8 * bound_x, y=+0.0 * bound_y, z=4.6 * bound_z),
                             carla.Rotation(pitch=6.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1.0, y=-1.0 * bound_y, z=0.4 * bound_z)), Attachment.Rigid)]

        self._camera_transforms[1] = (
            carla.Transform(
                carla.Location(x=0.5, y=0.0, z=2.3),  # מעט קדימה, על הגג
                carla.Rotation(pitch=-10.0, yaw=0.0, roll=0.0)  # מבט קדימה־למטה
            ),
            carla.AttachmentType.Rigid
        )

        self.transform_index = 1
        self.sensors = [['sensor.camera.rgb', cc.Raw, 'Camera RGB']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            bp.set_attribute('image_size_x', str(hud.dim[0]))
            bp.set_attribute('image_size_y', str(hud.dim[1]))
            bp.set_attribute('gamma', '2.2')
            item.append(bp)
        self.index = None
        self.setup_dual_cameras()

    def setup_dual_cameras(self):
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        WIDTH, HEIGHT = self.hud.dim

        transform = self._camera_transforms[self.transform_index][0]
        attach = self._camera_transforms[self.transform_index][1]

        # RGB Camera
        rgb_bp = bp_library.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', str(WIDTH))
        rgb_bp.set_attribute('image_size_y', str(HEIGHT))
        rgb_bp.set_attribute('fov', '120')
        rgb_bp.set_attribute('gamma', '2.2')
        self.sensor_rgb = world.spawn_actor(rgb_bp, transform, attach_to=self._parent, attachment_type=attach)
        self.sensor_rgb.listen(lambda image: self._parse_rgb_image(image))

        # Semantic Segmentation Camera
        seg_bp = bp_library.find('sensor.camera.semantic_segmentation')
        seg_bp.set_attribute('image_size_x', str(WIDTH))
        seg_bp.set_attribute('image_size_y', str(HEIGHT))
        seg_bp.set_attribute('fov', '120')
        self.sensor_seg = world.spawn_actor(seg_bp, transform, attach_to=self._parent, attachment_type=attach)
        self.sensor_seg.listen(lambda image: self._parse_seg_image(image))

    def _parse_rgb_image(self, image):

        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
        array = array[:, :, ::-1]  # BGR → RGB
        self.rgb_array = array.copy()

    def _parse_seg_image(self, image):

        self.process_start_time = time.time()
        # שליפת מזהים סמנטיים לפני ההמרה
        semantic_id_array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :,
                            2].copy()

        # המרה לצבעים
        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3].copy()

        # צביעת שמיים (ID=11) בתכלת
        sky_mask = (semantic_id_array == 11)
        array[sky_mask] = [135, 206, 250]  # light blue

        self.seg_array = array.copy()
        self.semantic_id_array = semantic_id_array

    def set_sensor(self, index, notify=True):
        pass  # אין תמיכה בהחלפת מצלמות, אז הפונקציה לא עושה כלום

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def monitor_bitrate(self):
        import re
        print("[bitrate monitor] started")
        while not self.bitrate_thread_stop and self.ffmpeg_stderr:
            try:
                line_bytes = self.ffmpeg_stderr.readline()
                if not line_bytes:
                    break
                decoded = line_bytes.decode("utf-8", errors="ignore")
                print(f"[ffmpeg stderr] {line_bytes}")  # DEBUG בלבד - אפשר למחוק

                if "bitrate=" in decoded:
                    match = re.search(r'bitrate=\s*(\d+\.?\d*)kbits/s', decoded)
                    if match:
                        bitrate_kbps = float(match.group(1))
                        print(f"[bitrate] {bitrate_kbps:.2f} kbps")
                        with open("bitrate_log.csv", "a") as f:
                            f.write(
                                f"{self.hud.frame},{round(time.time() - self.start_time, 2)},{bitrate_kbps * 1000:.2f}\n")
            except Exception as e:
                print(f"[bitrate monitor error] {e}")
                break


    def close_encoder(self):
        if self.out_file:
            self.out_file.close()
        if self.video_writer:
            self.video_writer.release()
        if self.sensor_rgb:
            self.sensor_rgb.destroy()
        if self.sensor_seg:
            self.sensor_seg.destroy()
        cv2.destroyAllWindows()

    def parse_nal_units(self, data: bytearray):
        """Parse NAL units and return their start positions and types."""
        nal_starts = []
        i = 0
        while i < len(data) - 4:
            if data[i:i + 4] in (b'\x00\x00\x00\x01', b'\x00\x00\x01'):
                start = i + (4 if data[i:i + 4] == b'\x00\x00\x00\x01' else 3)
                nal_type = (data[start] >> 1) & 0x3F  # HEVC NAL unit type
                nal_starts.append((start, nal_type))
                i = start
            else:
                i += 1
        return nal_starts

    def build_demuxer_and_decoder(self):
        self.feeder.reset(self.hdr_cache)
        try:
            self.demux = nvc.CreateDemuxer(self.feeder.feed)
            self.decoder = nvc.CreateDecoder(
                gpuid=self.gpu_id,
                codec=self.demux.GetNvCodecId(),
                outputColorType=nvc.OutputColorType.RGB,
                latency=nvc.DisplayDecodeLatencyType.ZERO
            )
            print("[DBG] Demuxer and Decoder created")
        except Exception as e:
            print(f"[ERROR] Demuxer/Decoder creation failed: {str(e)}")
            raise

    def show_decoded(self, surf, fid):
        buf = np.from_dlpack(surf)
        pitch_px = buf.size // (self.height * 3)
        frame = buf.reshape(self.height, pitch_px, 3)[:, :self.width, :].copy()
        cv2.putText(frame, f"f={fid}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow("Decoded Output", frame)
        print(f"[SHOW] Displaying frame {fid}, time={time.time():.3f}")
        if cv2.waitKey(1) == 27:
            self.sensor_rgb.stop()
            self.sensor_seg.stop()
            cv2.destroyAllWindows()
            self.out_file.close()
            if self.video_writer:
                self.video_writer.release()
            exit(0)

    def render(self, display):
        t0 = time.time()
        frame_id = self.hud.frame  # Use Carla's frame counter

        if self.rgb_array is not None and self.seg_array is not None and self.semantic_id_array is not None:
            ROI_IDS = [1, 2, 6, 7, 8, 12, 13, 14, 15, 16, 18, 24]
            '''
                        The next are ID for the object in Carla:
                        1 - road
                        2 - sidewalk
                        3 - buildings
                        4 - 
                        5 -
                        6 - lighting and electricity poles
                        7 - trafiic lights
                        8 - traffic signs
                        9 - shrubs, trees and vegetation
                        10 - 
                        11 - sky
                        ...
                        18 - motorcycles
                        19 - 
                        20 - 
                        21 - food track
                        22 - billboards/advertisements
                        23 - water
                        24 - pedestrian cross and traffic signs paint on the road
                        25 - soil
                        '''
            mask = np.isin(self.semantic_id_array, ROI_IDS)
            alpha = np.ones(mask.shape, dtype=np.float32)
            alpha[mask] = 0.0
            alpha = alpha[:, :, None]
            hybrid = (self.rgb_array * (1 - alpha) + self.seg_array * alpha).astype(np.uint8)

            # save raw video
            if not hasattr(self, "video_writer") or self.video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                fps = 20
                height, width = hybrid.shape[:2]
                self.video_writer = cv2.VideoWriter('hybrid_raw.avi', fourcc, fps, (width, height))
            bgr_hybrid = cv2.cvtColor(hybrid, cv2.COLOR_RGB2BGR)
            self.video_writer.write(bgr_hybrid)

            # GPU encode with NVENC
            b, g, r = cv2.split(hybrid)
            a = np.full_like(b, 255)
            argb = cv2.merge([b, g, r, a])
            pkt = self.encoder.Encode(argb)
            pkt2 = self.encoder.EndEncode()

            # Save packets for debugging
            if pkt:
                self.out_file.write(pkt)
            if pkt2:
                self.out_file.write(pkt2)

            # Prime header once
            if not self.got_header:
                for data in (pkt, pkt2):
                    if data:
                        self.hdr_cache.extend(data)
                if len(self.hdr_cache) > 40000:
                    self.got_header = True
                    print("[HDR] Header cached (~{} bytes)".format(len(self.hdr_cache)))
                    nal_starts = self.parse_nal_units(self.hdr_cache)
                    print(f"[HDR] Found NAL units: {[(pos, typ) for pos, typ in nal_starts]}")
                    end_pos = len(self.hdr_cache)
                    for i, (pos, nal_type) in enumerate(nal_starts):
                        if nal_type == 19 and i >= 3:  # First IDR after VPS/SPS/PPS
                            end_pos = nal_starts[i + 1][0] if i + 1 < len(nal_starts) else len(self.hdr_cache)
                            break
                    self.hdr_cache = self.hdr_cache[:end_pos]
                    print("[HDR] Trimmed to ~{} bytes (VPS/SPS/PPS/IDR)".format(len(self.hdr_cache)))
                    with open("hdr_cache.h265", "wb") as f:
                        f.write(self.hdr_cache)
                    self.build_demuxer_and_decoder()
            else:
                # Feed new packets
                if pkt:
                    self.feeder.append(pkt)
                if pkt2:
                    self.feeder.append(pkt2)

                # Decode packets (process one packet per frame)
                read_any = False
                for p in self.demux:
                    read_any = True
                    p.decode_flag = nvc.VideoPacketFlag.ENDOFPICTURE
                    #print(f"[DECODE] Processing packet, size={len(p.data)}")
                    surfaces = list(self.decoder.Decode(p))
                    print(f"[DECODE] Got {len(surfaces)} surfaces")
                    for surf in surfaces:
                        self.show_decoded(surf, frame_id)
                        time.sleep(0.001)  # Small delay for zero-latency stability
                    break  # Process one packet per frame

                # Handle demuxer stall
                if not read_any and self.feeder.has_data():
                    print("[DBG] Demuxer stalled, rebuilding...")
                    leftover = self.feeder.buf[self.feeder.pos:]
                    print(f"[DBG] Leftover data: {len(leftover)} bytes")
                    self.feeder.clear()
                    self.feeder.reset(self.hdr_cache + leftover)
                    self.build_demuxer_and_decoder()

                # Consume processed data
                if read_any:
                    self.feeder.consume(self.feeder.pos)

            # Measure latency
            if self.process_start_time is not None:
                latency_ms = (time.time() - self.process_start_time) * 1000
                self.frame_times.append(latency_ms)
                average_time_ms = sum(self.frame_times) / len(self.frame_times)
                with open("hybrid_latency.csv", "a") as f:
                    f.write(f"{frame_id},{latency_ms:.2f},{average_time_ms:.2f}\n")

            # Pygame display
            surface = pygame.surfarray.make_surface(hybrid.swapaxes(0, 1))
            display.blit(surface, (0, 0))

            # Maintain FPS
        elapsed = time.time() - t0
        time.sleep(max(0.0, self.dt - elapsed))




# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)
        sim_world = client.get_world()

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0,0,0))
        pygame.display.flip()

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world, args.autopilot)

        sim_world.wait_for_tick()

        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(client, world, clock):
                return
            if not world.tick(clock, args.wait_for_repetitions):
                return
            world.render(display)
            pygame.display.flip()

    finally:

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            # prevent destruction of ego vehicle
            if args.keep_ego_vehicle:
                world.player = None
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot. This does not autocomplete the scenario')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='role name of ego vehicle to control (default: "hero")')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='640x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--keep_ego_vehicle',
        action='store_true',
        help='do not destroy ego vehicle on exit')
    argparser.add_argument(
        '--wait-for-repetitions',
        action='store_true',
        help='Avoids stopping the manual control when the scenario ends.')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except Exception as error:
        logging.exception(error)


if __name__ == '__main__':

    main()
