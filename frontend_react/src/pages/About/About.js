import React from "react";

import "./About.css";

export default function About() {
  return (
    <div className="about-container">
      <div className="about-header">
        <h1 className="about-heading">About</h1>
      </div>
      <div className="about-main">
        <p className="about-content">
          This is an realtime AI based Yoga Trainer which detects your pose how
          well you are doing. This AI first predicts keypoints or coordinates of
          different parts of the body(basically where they are present in an
          image) and then it use another classification model to classify the
          poses if someone is doing a pose and if AI detects that pose more than
          95% probability and then it will notify you are doing correctly(by
          making virtual skeleton green). We have used Tensorflow pretrained
          Movenet Model To Predict the Keypoints and building a neural network
          top of that which uses these coordinates and classify a yoga pose. We
          have trained the model in python because of tensorflowJS we can
          leverage the support of browser so we converted the keras/tensorflow
          model to tensorflowJS. <br />
          We are looking at the following improvements:
          <ul>
            <li>Voice based correction system</li>
            <li>Inclusion of more Yogic Poses</li>
          </ul>
        </p>
        <div className="developer-info">
          <h4>About Developers</h4>
          <p className="about-content">
            We are Aashish C, Deekshajyothi S and B Shrikara. <br />
            We are final year engineering students at BMS College of Engineering
            and this is created as apart of our final year project.
          </p>
        </div>
      </div>
    </div>
  );
}
