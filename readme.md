# Pulmoscope AI: Robust Cross-Dataset Lung & Nodule Segmentation

A lightweight, multi-class semantic segmentation framework designed to identify lungs, tiny nodules (3mm), and infections (GGO/Consolidation) across highly heterogeneous CT imaging datasets. 

This project tackles three major clinical machine learning challenges: preserving high-resolution spatial data for micro-nodules, eliminating cross-dataset scanner bias, and providing pixel-wise confidence metrics for clinical practitioners. Optimized for memory-constrained hardware (e.g., 4GB VRAM GPUs like the RTX 3050), it achieves state-of-the-art performance without requiring massive compute clusters.

## 🚀 Key Innovations
*   **Multi-Scale Dilated Encoder:** Modifies a standard ResNet18 backbone with dilated convolutions to prevent spatial resolution loss deep in the network.
*   **Hybrid Attention (CBAM + FFT):** Combines spatial/channel attention with Fast Fourier Transform (FFT) frequency-domain filtering to isolate sharp nodule boundaries.
*   **Adaptive Instance Normalization (AdaIN):** Normalizes feature maps to eliminate domain shift (scanner bias) when mixing LUNA16 and COVID-19 datasets.
*   **Monte Carlo Dropout:** Runs 10 inference passes to generate clinical Uncertainty Heatmaps, acting as a "safety net" against false positives.

## 📊 Dataset & Class Schema
The model was trained on 6,953 high-resolution CT slices across 3,609 unique patients, utilizing a unified 4-class mask schema[cite: 1]:
*   **0 - Background:** Standard background[cite: 1].
*   **1 - Lung:** Healthy lung tissue derived from LUNA16 and COVID datasets[cite: 1].
*   **2 - Nodule:** 3D spherical nodule representations derived from LUNA16 annotations[cite: 1].
*   **3 - Infection:** Unified Ground Glass Opacity (GGO) and Consolidation[cite: 1].

## 🧠 Architecture Overview

```mermaid
graph TD
    A[Input CT Slice] --> B[Module 1: Dilated ResNet18 Encoder]
    B --> C[Module 2: Hybrid Attention CBAM + FFT]
    
    subgraph Domain Alignment
    C --> D[Module 3: AdaIN Normalization]
    end
    
    D --> E[Module 5: Monte Carlo Dropout p=0.3]
    E --> F[Module 4: Lightweight Decoder]
    F --> G([Mean Segmentation Mask])
    
    E -.->|10 Forward Passes| H([Uncertainty Heatmap])
    
    classDef output fill:#f9f,stroke:#333,stroke-width:2px;
    class G,H output;