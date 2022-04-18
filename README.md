# COVID-19-Diagnosis-Model-External-Validation-Set

This <a href="https://drive.google.com/file/d/1elZK-JVWuJuvMs95Ott2leAP5DJtVlZE/view?usp=sharing"> [dataset]</a>  is a benchmark for external validation of automatic COVID-19 diangosis models. It includes 1163 images of COVID-19 positive cases and 1349 images of COVID-19 negative cases. Images are converted to 8bits and lungs are segmented and cropped.

The pre-trained Segmentation U-net model can be downloaded from <a href="https://drive.google.com/file/d/14brQsEFOJOKCo0qg0DXhfGVSf3RACojX/view?usp=sharing">[here]</a>

For model training use: train.py

The original Sources used in this set are :

COVID-19 Positive:
BIMCV-COVID-19+ <a href="https://arxiv.org/abs/2006.01174">[(Spain)]</a>, COVID-19-AR <a href="10.7937/tcia.2020.py71-5978">[(USA)]</a> , CC-CXRI-P<a href=""> [(China)]</a>, <a href="10.6084/m9.figshare.12275009.v1">[V2-COV19-NII]</a> (Germany)  and COVIDGR (Spain-Granada) . 
COVID-19 Negative:
the National Institutes of Health (NIH) Chest Xray dataset, Chexpert  and <a href="10.1016/j.media.2020.101797">[Padchest]</a>. 


For questions, email us at: kbenahmed@usf.edu





