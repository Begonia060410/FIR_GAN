# FIR_GAN: High-Fidelity Multimodal Image Fusion with Frequency-Domain Attention for Robust Visual Detection

High-Fidelity Multimodal Image Fusion with Frequency-Domain Attention for Robust Visual Detection

Multimodal image fusion integrates complementary information across modalities to improve the reliability of high-level vision tasks such as object detection. However, heterogeneous distributions, misalignment artifacts, and inefficient training limit fusion quality and downstream performance. This work introduces a high-fidelity fusion framework dedicated to detection scenarios, featuring a dual-branch CNN-Transformer architecture and a statistical frequency-domain attention mechanism to enhance feature alignment and detail preservation. A block-level index-guided optimization strategy is proposed to reduce computational cost while stabilizing training. Here we show the method improves VIF by 0.28 on TNO and mAP@0.5 by 10.3% on DroneVehicle compared to state-of-the-art approaches. The framework effectively balances fusion fidelity and efficiency, supporting robust cross-modal perception for real-world vision systems.


<img width="2431" height="563" alt="1" src="https://github.com/user-attachments/assets/71cd1c81-fb73-4801-b82c-2eb70c01dda0" />
Fig. 1 Visual comparison of different image fusion algorithms
<img width="1646" height="993" alt="2" src="https://github.com/user-attachments/assets/06dc572b-0b9d-4fab-8d91-7f1485b050a9" />
Fig. 2 Conceptual framework of the proposed FIR-GAN.(a) Full architecture of the system.(b) Patch-level IQA index–guided training optimization strategy.(c) Structure of the generator network.(d) Structure of the discriminator network.



A permanent archived version with DOI is available at: https://doi.org/10.5281/zenodo.20289332.

If this work has been helpful to you, please cite the following LaTeX source code.

@article{FIRGAN2026,

title={High-Fidelity Multimodal Image Fusion with Frequency-Domain Attention for Robust Visual Detection},

journal={The Visual Computer},

year={2026}

}
