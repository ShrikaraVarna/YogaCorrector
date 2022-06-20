import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs";
import React, { useRef, useState, useEffect } from "react";
import Webcam from "react-webcam";
import { count } from "../../utils/music";
import { Alert, Dialog } from "@mui/material";
import Typography from "@mui/material/Typography";
import Speech from "react-speech";

import Instructions from "../../components/Instrctions/Instructions";

import "./Yoga.css";

import DropDown from "../../components/DropDown/DropDown";
import { poseImages } from "../../utils/pose_images";
import { POINTS, keypointConnections } from "../../utils/data";
import { drawPoint, drawSegment } from "../../utils/helper";

let skeletonColor = "rgb(255,255,255)";
let poseList = [
  "Adho_Mukha_Svanasana",
  "Adho_Mukha_Vriksasana",
  "Bhekasana",
  "Bhunjangasana",
  "Chakravakasana",
  "Eka_Pada_Koundinyanasana_I",
  "Padmasana",
  "Simhasana",
  "Tadasana",
  "Trikonasana",
  "Virabhadrasana_I",
  "Virabhadrasana_II",
  "Virabhadrasana_III",
];

let interval;
let text = "";
let url = "http://127.0.0.1:8000/json?finaldata=";

// flag variable is used to help capture the time when AI just detect
// the pose as correct(probability more than threshold)
let flag = false;

function Yoga() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  const [poseDone, setPoseDone] = React.useState("");

  //const [text, setText] = React.useState("");

  const [accuracy, setAccuracy] = React.useState(0);
  const [alertOpen, setAlertOpen] = React.useState(false);
  const [accuracyvalue, showAccuracyValue] = React.useState(false);
  const [startingTime, setStartingTime] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [poseTime, setPoseTime] = useState(0);
  const [bestPerform, setBestPerform] = useState(0);
  const [currentPose, setCurrentPose] = useState("Tadasana");
  const [isStartPose, setIsStartPose] = useState(false);
  const [allSlopes, setAllSLopes] = useState({});

  const getData = () => {
    fetch("slopes.json", {
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
    })
      .then(function (response) {
        return response.json();
      })
      .then(function (myJson) {
        setAllSLopes(myJson);
      });
  };

  useEffect(() => {
    getData();
  }, []);

  useEffect(() => {
    const timeDiff = (currentTime - startingTime) / 1000;
    if (flag) {
      setPoseTime(timeDiff);
    }
    if ((currentTime - startingTime) / 1000 > bestPerform) {
      setBestPerform(timeDiff);
    }
  }, [currentTime]);

  useEffect(() => {
    setCurrentTime(0);
    setPoseTime(0);
    setBestPerform(0);
  }, [currentPose]);

  const CLASS_NO = {
    Adho_Mukha_Svanasana: 0,
    Adho_Mukha_Vriksasana: 1,
    Bhekasana: 2,
    Bhunjangasana: 3,
    Chakravakasana: 4,
    Eka_Pada_Koundinyanasana_I: 5,
    Padmasana: 6,
    Simhasana: 7,
    Tadasana: 8,
    Trikonasana: 9,
    Virabhadrasana_I: 10,
    Virabhadrasana_II: 11,
    Virabhadrasana_III: 12,
  };

  const Pose_NO = {
    0: "Adho_Mukha_Svanasana",
    1: "Adho_Mukha_Vriksasana",
    2: "Bhekasana",
    3: "Bhunjangasana",
    4: "Chakravakasana",
    5: "Eka_Pada_Koundinyanasana_I",
    6: "Padmasana",
    7: "Simhasana",
    8: "Tadasana",
    9: "Trikonasana",
    10: "Virabhadrasana_I",
    11: "Virabhadrasana_II",
    12: "Virabhadrasana_III",
  };

  function get_center_point(landmarks, left_bodypart, right_bodypart) {
    let left = tf.gather(landmarks, left_bodypart, 1);
    let right = tf.gather(landmarks, right_bodypart, 1);
    const center = tf.add(tf.mul(left, 0.5), tf.mul(right, 0.5));
    return center;
  }

  const check_if_arms_straight = (keypoints, currentPose) => {
    let left_slope =
      (keypoints[9].y - keypoints[5].y) / (keypoints[9].x - keypoints[5].x);
    let right_slope =
      (keypoints[10].y - keypoints[6].y) / (keypoints[10].x - keypoints[6].x);
    if (
      allSlopes[currentPose].right_arm_min > right_slope ||
      allSlopes[currentPose].right_arm_max < right_slope
    ) {
      let newText = text + " Your right arm isn't as it should be.";
      text = newText;
    }
    if (
      allSlopes[currentPose].left_arm_min > left_slope ||
      allSlopes[currentPose].left_arm_max < left_slope
    ) {
      let newText =
        text + " Try to maintain your left arm as shown in the image.";
      //setText(newText);
      text = newText;
    }
  };

  const check_if_right_knee_straight = (keypoints, currentPose) => {
    let slope =
      (keypoints[12].y - keypoints[16].y) / (keypoints[12].x - keypoints[16].y);
    if (slope > 0.4) {
      let newText = text + " Your right isn't aligned properly.";
      //setText(newText);
      text = newText;
    }
  };

  const check_if_left_knee_straight = (keypoints, currentPose) => {
    let slope =
      (keypoints[11].y - keypoints[15].y) / (keypoints[11].x - keypoints[15].y);
    if (slope > 0.4) {
      let newText = text + " Your left knee is also not properly placed.";
      //setText(newText);
      text = newText;
    }
  };

  function get_pose_size(landmarks, torso_size_multiplier = 2.5) {
    let hips_center = get_center_point(
      landmarks,
      POINTS.LEFT_HIP,
      POINTS.RIGHT_HIP
    );
    let shoulders_center = get_center_point(
      landmarks,
      POINTS.LEFT_SHOULDER,
      POINTS.RIGHT_SHOULDER
    );
    let torso_size = tf.norm(tf.sub(shoulders_center, hips_center));
    let pose_center_new = get_center_point(
      landmarks,
      POINTS.LEFT_HIP,
      POINTS.RIGHT_HIP
    );
    pose_center_new = tf.expandDims(pose_center_new, 1);

    pose_center_new = tf.broadcastTo(pose_center_new, [1, 17, 2]);
    // return: shape(17,2)
    let d = tf.gather(tf.sub(landmarks, pose_center_new), 0, 0);
    let max_dist = tf.max(tf.norm(d, "euclidean", 0));

    // normalize scale
    let pose_size = tf.maximum(
      tf.mul(torso_size, torso_size_multiplier),
      max_dist
    );
    return pose_size;
  }

  function normalize_pose_landmarks(landmarks) {
    let pose_center = get_center_point(
      landmarks,
      POINTS.LEFT_HIP,
      POINTS.RIGHT_HIP
    );
    pose_center = tf.expandDims(pose_center, 1);
    pose_center = tf.broadcastTo(pose_center, [1, 17, 2]);
    landmarks = tf.sub(landmarks, pose_center);

    let pose_size = get_pose_size(landmarks);
    landmarks = tf.div(landmarks, pose_size);
    return landmarks;
  }

  function landmarks_to_embedding(landmarks) {
    // normalize landmarks 2D
    landmarks = normalize_pose_landmarks(tf.expandDims(landmarks, 0));
    let embedding = tf.reshape(landmarks, [1, 34]);
    return embedding;
  }

  const runMovenet = async () => {
    const detectorConfig = {
      modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER,
    };
    const detector = await poseDetection.createDetector(
      poseDetection.SupportedModels.MoveNet,
      detectorConfig
    );
    const poseClassifier = await tf.loadLayersModel("model/model.json");
    const countAudio = new Audio(count);
    countAudio.loop = true;
    interval = setInterval(() => {
      detectPose(detector, poseClassifier, countAudio);
    }, 100);
  };

  const detectPose = async (detector, poseClassifier, countAudio) => {
    let nd = [];
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      let notDetected = 0;
      const video = webcamRef.current.video;
      const pose = await detector.estimatePoses(video);
      const ctx = canvasRef.current.getContext("2d");
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      try {
        const keypoints = pose[0].keypoints;
        let input = keypoints.map((keypoint) => {
          if (keypoint.score > 0.4) {
            if (
              !(keypoint.name === "left_eye" || keypoint.name === "right_eye")
            ) {
              drawPoint(ctx, keypoint.x, keypoint.y, 8, "rgb(255,255,255)");
              let connections = keypointConnections[keypoint.name];
              try {
                connections.forEach((connection) => {
                  let conName = connection.toUpperCase();
                  drawSegment(
                    ctx,
                    [keypoint.x, keypoint.y],
                    [
                      keypoints[POINTS[conName]].x,
                      keypoints[POINTS[conName]].y,
                    ],
                    skeletonColor
                  );
                });
              } catch (err) {}
            }
          } else {
            nd = [...nd, keypoint];
            notDetected += 1;
          }
          return [keypoint.x, keypoint.y];
        });
        if (notDetected > 4) {
          text = "";
          setAlertOpen(false);
          skeletonColor = "rgb(255,0,0)";
          return;
        }
        const processedInput = landmarks_to_embedding(input);
        const classification = poseClassifier.predict(processedInput);

        classification.array().then((data) => {
          let highestClass = 20;
          let highestScore = 0;
          let clNo = -1;
          data[0].map((pose) => {
            clNo = clNo + 1;
            if (pose > highestScore) {
              highestScore = pose;
              highestClass = clNo;
            }
          });
          if (highestClass !== CLASS_NO[currentPose]) {
            setAlertOpen(!alertOpen);
            showAccuracyValue(false);
            setPoseDone(Pose_NO[highestClass]);
            skeletonColor = "rgb(0,0,0)";
            text = "";
            countAudio.pause();
          } else {
            setAlertOpen(false);
            const classNo = CLASS_NO[currentPose];
            if (data[0][classNo] > 0.95) {
              if (!flag) {
                setAccuracy(data[0][classNo] * 100);
                showAccuracyValue(true);
                countAudio.play();
                setStartingTime(new Date(Date()).getTime());
                flag = true;
                setCurrentTime(new Date(Date()).getTime());
                skeletonColor = "rgb(0,255,0)";
                check_if_arms_straight(keypoints, currentPose);
                check_if_left_knee_straight(keypoints, currentPose);
                check_if_right_knee_straight(keypoints, currentPose);
                let newText = text + " Please see the image being displayed.";
                text = newText;
                url = url + text;
                console.log(newText);
                //setText(newText);
              }
              setCurrentTime(new Date(Date()).getTime());
            } else {
              text = "";
              flag = false;
              skeletonColor = "rgb(0,0,255)";
              countAudio.pause();
              countAudio.currentTime = 0;
            }
          }
        });
      } catch (err) {
        console.log(err);
      }
    }
  };

  function startYoga() {
    setIsStartPose(true);
    runMovenet();
    <Speech text={text} />;
  }

  function stopPose() {
    setIsStartPose(false);
    clearInterval(interval);
  }

  if (isStartPose) {
    return (
      <div className="yoga-container">
        <div className="performance-container" style={{ top: 0, padding: 0 }}>
          <div className="pose-performance">
            <h4>Pose Time: {poseTime} s</h4>
          </div>
          <div className="pose-performance">
            <h4>Best: {bestPerform} s</h4>
          </div>
        </div>
        <div>
          <Webcam
            width="640px"
            height="480px"
            id="webcam"
            ref={webcamRef}
            style={{
              position: "absolute",
              left: 120,
              top: 360,
              padding: "0px",
            }}
          />
          <canvas
            ref={canvasRef}
            id="my-canvas"
            width="640px"
            height="480px"
            style={{
              position: "absolute",
              left: 120,
              top: 360,
              zIndex: 1,
            }}
          ></canvas>
          <div
            style={{
              position: "absolute",
              right: "10px",
              top: 300,
              zIndex: 1,
            }}
          >
            <img src={poseImages[currentPose]} className="pose-img" />
          </div>
        </div>
        <Dialog open={alertOpen}>
          <Alert severity="error" onClose={() => setAlertOpen(!alertOpen)}>
            You are doing a different Pose. The pose done by you is {poseDone}
          </Alert>
        </Dialog>
        <button onClick={stopPose} className="secondary-btn">
          Stop Pose
        </button>
        {accuracyvalue ? (
          <div
            style={{
              top: "30px",
              textAlign: "center",
              alignItems: "center",
            }}
          >
            <Typography variant="h1" component="h3">
              Your accuracy is {accuracy}.
            </Typography>
            <Typography variant="h3" component="h3">
              {text}
            </Typography>
          </div>
        ) : null}
        <Speech text={text} voice="Google UK English Female" />
      </div>
    );
  }

  return (
    <div className="yoga-container">
      <DropDown
        poseList={poseList}
        currentPose={currentPose}
        setCurrentPose={setCurrentPose}
      />
      <Instructions currentPose={currentPose} />
      <button onClick={startYoga} className="secondary-btn">
        Start Pose
      </button>
    </div>
  );
}

export default Yoga;
