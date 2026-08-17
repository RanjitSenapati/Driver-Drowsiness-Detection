# Model File Required

Download the dlib 68-point facial landmark predictor (not included here due to
GitHub/file-size limits) and place it in this folder:

```bash
curl -L -o shape_predictor_68_face_landmarks.dat.bz2 \
  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

The final file should be at: `models/shape_predictor_68_face_landmarks.dat`
