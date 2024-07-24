# DeepSOZ
DeepSOZ: A Robust Deep Model for Joint Temporal and Spatial Seizure Onset Localization from Multichannel EEG Data

<h2> Abstract </h2>
We propose a robust deep learning framework to simultaneously detect and localize seizure activity from multichannel scalp EEG.Our model, called DeepSOZ, consists of a transformer encoder to generate global and channel-wise encodings. The global branch is combined with an LSTM for temporal seizure detection. In parallel, we employ attention-weighted multi-instance pooling of channel-wise encodings to predict the seizure onset zone. DeepSOZ is trained in a supervised fashion and generates high-resolution predictions on the order of each second (temporal) and EEG channel (spatial). We validate DeepSOZ via bootstrapped nested cross-validation on a large dataset of 120 patients curated from the Temple University Hospital corpus. As compared to baseline approaches, DeepSOZ provides robust overall performance in our multi-task learning setup. We also evaluate the intra-seizure and intra-patient consistency of DeepSOZ as a first step to establishing its trustworthiness for integration into the clinical workflow for epilepsy.

<h2> Data Access </h2>
<p>Download TUH data from: <a href="https://isip.piconepress.com/projects/tuh_eeg/">TUH Seizure Corpus</a></p>
Please follow the instructions on the website for access and downloading. In this work, we only use the Seizure subcorpus. 

<h2> Cite as </h2>
<p> M. Shama, D., Jing, J., Venkataraman, A. (2023). DeepSOZ: A Robust Deep Model for Joint Temporal and Spatial Seizure Onset Localization from Multichannel EEG Data. In: Greenspan, H., et al. Medical Image Computing and Computer Assisted Intervention – MICCAI 2023. MICCAI 2023. Lecture Notes in Computer Science, vol 14227. Springer, Cham. https://doi.org/10.1007/978-3-031-43993-3_18 </p>

