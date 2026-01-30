# Studying the Effectiveness of Hybrid Video Compression for Teleoperated Driving

This project explores a hybrid video compression method designed to optimize real-time video transmission for remote driving (Teleoperation) of autonomous vehicles. By combining Semantic Segmentation for non-critical regions (non-ROI) and high-quality Photorealistic RGB for critical regions (ROI), we achieve significant bandwidth reduction while maintaining the driver's spatial awareness.




Click the image below to watch the system in action, demonstrating the hybrid compression and the cyclist scenario:

[![Hybrid Video Compression Demo](https://img.youtube.com/vi/s7sZhRlwCF0/0.jpg)](https://youtu.be/s7sZhRlwCF0)




## Key Features

Hybrid Image Generation: Merges RGB and semantic segmentation frames in real-time using GPU-accelerated processing (CuPy).

Real-time Hardware Compression: Utilizes NVIDIA NVENC/NVDEC (H.265/HEVC) to maintain low latency (avg. 44.68ms processing time).

Bandwidth Optimization: Optimized for 4G/5G networks, maintaining a stable bitrate of ~898 kbps (target 1 Mbps).

CARLA Integration: Built on top of the CARLA simulator with custom driving scenarios.

## Tech Stack

**Simulator:** CARLA

**Libraries:** Python (API), PyNvVideoCodec (NVIDIA), CuPy (GPU processing), PyGame (UI & Control).

**Codec:** H.265/HEVC.

**Hardware:** NVIDIA RTX-5060 GPU.

## Methodology

The system captures two synchronized streams: a photorealistic RGB frame and a semantic segmentation frame. It creates a "Hybrid Frame" where non-ROI objects (like sky and buildings) are simplified into uniform colors based on the Cityscapes labeling system, while the ROI (road, vehicles, pedestrians) remains high quality.

## Final Research Stages

The project's execution is divided into two final critical stages, representing the full cycle of the hybrid compression pipeline:

**Stage A:** Real-Time Hybrid Encoding

Core Script: manual_control_enc_dec_PyNvVideoCodec.py

Process: This stage implements the heart of our research. It captures the simulation data, generates the hybrid frame (RGB + Semantic), and utilizes hardware-accelerated encoding via NVIDIA NVENC. The goal is to achieve maximum compression without sacrificing the low-latency requirements of teleoperation.

**Stage B:** Decoding & Visual Validation

Core Script: manual_control_decoded.py

Process: This is the "User End" of the system. The stream is decoded using NVDEC and displayed in a wide-screen Side-by-Side format (2560x720). This allows for a direct comparison between the raw CARLA output and our hybrid-compressed stream, facilitating the PSNR and bitrate analysis presented in the final report.

## Acknowledgments & Credits

This project is built upon the following open-source tools:

**CARLA Simulator:** Used as the primary simulation environment.

**Scenario Runner:** This project leverages the scenario_runner framework. We specifically modified the manual_control.py script to integrate our hybrid compression pipeline and custom teleoperation controls.

## Final Report

For more details regarding the research, experiment results, and technical architecture, please refer to the Full Project Report.

## Using

### 1. Prerequisites
* **CARLA Simulator:** Download and install **CARLA 0.9.15**.
* **Anaconda:** Required for environment management.
* **NVIDIA GPU:** Required for hardware-accelerated video encoding/decoding.

### Environment Setup
1. Clone the official [Scenario Runner](https://github.com/carla-simulator/scenario_runner) (v0.9.15 compatible).
2. Place the project file `manual_control_enc_dec_PyNvVideoCodec.py` and environment.yml inside the root folder of the Scenario Runner.
3. Open Anaconda Prompt, navigate to the folder, and run:
   ```bash
   conda env create -f environment.yml
   conda activate carla-sim
> [!NOTE]
         A system restart might be required after installation.

### 3. Path Configuration

If you encounter "Agent" or "Carla" module errors, manually set the PYTHONPATH:

<pre>
set PYTHONPATH=%PYTHONPATH%;C:\Path\To\Your\CARLA\WindowsNoEditor\PythonAPI\carla
</pre>

### How to Run (Step-by-Step)

Follow these steps using three separate terminal tabs:
**Step 1:** Start CARLA

Navigate to your CARLA folder and run:

    CarlaUE4.exe -windowed -ResX=800 -ResY=600

To change maps: Press ~ in the simulator and type open Town01.

**Step 2:** Run the Hybrid Control Script

In a new terminal (activated with conda activate carla-sim):

      python manual_control_enc_dec_PyNvVideoCodec.py
      python manual_control_decoded.py

Two windows will appear: One showing the compressed hybrid video and one showing the raw stream.

<img width="1340" height="742" alt="image25" src="https://github.com/user-attachments/assets/8f754f9d-5ea8-4857-af4b-fc9806ac116b" />

**Step 3:** Launch a Scenario

In a third terminal (activated with conda activate carla-sim):

View available scenarios: 
    python scenario_runner.py --list

Run a scenario:
    
    python scenario_runner.py --scenario ChangeLane_1

<img width="1280" height="720" alt="image30" src="https://github.com/user-attachments/assets/593e05e5-d79d-4b82-ab5b-eecefeb51d94" />

<img width="1280" height="720" alt="image24" src="https://github.com/user-attachments/assets/d181a3b6-0725-45a2-a2fc-976ffc022bdb" />

## Acknowledgments

This project is an extension of the CARLA Scenario Runner. We have modified the manual_control.py logic to implement our hybrid compression research.
