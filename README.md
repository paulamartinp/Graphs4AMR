## Advanced Graph-Based Approaches for Predicting Antimicrobial Resistance in Intensive Care Units

Paula Martín-Palomeque, Óscar Escudero-Arnanz, Cristina Soguero-Ruiz, Antonio G. Marques

Dept. Signal Theory & Communications, Telematics & Computing Systems,
Rey Juan Carlos University, Fuenlabrada, Spain


### Abstract
Antimicrobial Resistance (AMR) poses a signifi-
cant global public health challenge, necessitating early detection strategies to enable timely clinical interventions. Electronic Health Records (EHRs) offer extensive real-world clinical data but present challenges due to their irregularly sampled, heterogeneous, and multivariate temporal structure. This paper investigates graph-based deep learning models to predict AMR in Intensive Care Unit (ICU) patients by systematically modeling spatial and temporal dependencies within EHR data represented as Multivariate Time Series (MTS). We propose and evaluate
a novel Spatio-Temporal Graph Convolutional Neural Network (ST-GCNN) architecture, demonstrating its superior predictive
performance by achieving a Receiver Operating Characteristic
Area Under the Curve (ROC AUC) of 80.00%, surpassing baseline models by approximately 6%. Furthermore, our analysis of the learned graph structures highlights critical clinical interactions, notably emphasizing catheter-related variables as
central nodes, aligning well with established clinical knowledge. By combining high predictive performance with enhanced interpretability, our approach presents a robust and transparent framework, well-suited for clinical applications aimed at improving AMR risk assessment and patient care management.


### Repository overiew
- **Baselines/** 
    - Baseline models for comparison
- **E0_graphEstimation/** 
    - Graph estimation and adjacency matrix construction
- **E1_InferenceOverGraph/**    
    - Inference and node analysis over estimated graphs
- **E2_Standard-GCN/**
    - Implementation of standard Graph Convolutional Networks
- **E3_SpatioTemporal-GCN_HighOrder/**
    - Proposed ST-GCNN architecture and experiments
- **utils.py**  
    - Utility functions
