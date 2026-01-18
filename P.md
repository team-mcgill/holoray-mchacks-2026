
HoloRay Challenge: Motion-Tracked Annotation Solution 

The Challenge: 
HoloXR is a collaborative, real-time collaboration platform designed for healthcare providers. Users can currently draw 2D annotations over live video sources—similar to a video conferencing environment—but these annotations are static and do not move with the underlying imagery. This limitation becomes particularly evident when annotating medical imaging feeds such as ultrasound, echocardiography, laparoscopy, IVUS, and other diagnostic or procedural video sources, where camera motion and anatomical motion are common. 
The challenge is to develop a motion-tracked annotation solution that keeps annotations fixed relative to the anatomical feature they were originally placed on. For example, if an ultrasound probe shifts left by 2 cm, the annotation should automatically move left by 2 cm on screen, remaining aligned with the same anatomy. 
Participants are free to choose their technical approach, including but not limited to: 
Classical computer vision methods (e.g., OpenCV feature matching or object trackers) 
Optical flow–based tracking 
Deep learning or hybrid approaches 
Any other novel or experimental techniques 
Key considerations include: 
Real-time performance: Teams should define what “real-time” means for their solution and justify it (for example, by specifying acceptable latency thresholds or response times). 
Sampling and frame rate: Tracking should ideally operate at or near the video source FPS to ensure smooth and accurate annotation movement. 
Robustness: Handling partial occlusion, anatomical motion, and cases where annotated regions move out of frame and later re-enter. 
Generality: Applicability across different medical video modalities. 
HoloRay will provide sample medical videos (e.g., echocardiography, laparoscopy), but participants may use additional publicly available videos if they align with the intended clinical use case. 
Participants may optionally extend their solution by building a minimal web UI and integrating their tracking system into a basic collaborative video setup using open-source WebRTC tooling (cloud-based or peer-to-peer). 

Judging Criteria:
Submissions will be evaluated across the following dimensions: 
Tracking Accuracy & Stability 
How well annotations remain anchored to the intended anatomical structures 
Resistance to drift, jitter, or misalignment during motion 
Real-Time Performance 
Latency and responsiveness of the tracking solution 
Alignment between tracking updates and video frame rate 
Robustness & Edge Case Handling 
Performance during camera motion, anatomy deformation, occlusion, and out-of-frame scenarios 
Recovery when tracked anatomy re-enters the frame 
Technical Approach & Innovation
Soundness of the chosen methodology 
Creativity or novelty in combining techniques or solving known challenges 
Bonus (Optional Enhancements) 
Functional UI or WebRTC integration 
Clear visualization of annotations in a collaborative or multi-user context 

Prizes:
$300 CAD per participant for each member of the winning team 
Fast-track to final-round interview consideration for a Summer Internship at HoloRay 

Dataset: 
https://we.tl/t-dQX1Q5lAf0