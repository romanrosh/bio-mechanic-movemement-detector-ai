I also get confused. MPII datasets has 16 keypoints
(0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist).
However, I find this code in modelDescriptorFactory.cpp
{{0, "Head"}, //head top
{1, "Neck"}, //upper neck
{2, "RShoulder"}, //r shoulder
{3, "RElbow"}, // r elbow
{4, "RWrist"}, // r wrist
{5, "LShoulder"}, // l shoulder
{6, "LElbow"}, // l elbow
{7, "LWrist"}, // l wrist
{8, "RHip"}, //r hip
{9, "RKnee"}, // r knee
{10, "RAnkle"}, //r ankle
{11, "LHip"}, //l hip
{12, "LKnee"}, //l knee
{13, "LAnkle"},//l ankle
{14, "Chest"},// thorax
{15, "Bkg"}}, // ?? pelvis
Does "Bkg" represent background? Where is the pelvis?

And coco has 17 keypoints which are:
['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'].
but your code:
{{0, "Nose"}, //t
{1, "Neck"}, //f is not included in coco. How do you get the Neck keypoints
{2, "RShoulder"}, //t
{3, "RElbow"}, //t
{4, "RWrist"}, //t
{5, "LShoulder"}, //t
{6, "LElbow"}, //t
{7, "LWrist"}, //t
{8, "RHip"}, //t
{9, "RKnee"}, //t
{10, "RAnkle"}, //t
{11, "LHip"}, //t
{12, "LKnee"}, //t
{13, "LAnkle"}, //t
{14, "REye"}, //t
{15, "LEye"}, //t
{16, "REar"}, //t
{17, "LEar"}, //t
{18, "Bkg"}}, //f background ??