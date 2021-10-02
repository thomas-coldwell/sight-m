import { StatusBar } from "expo-status-bar";
import React, { useRef, useState } from "react";
import {
  ActivityIndicator,
  Button,
  Dimensions,
  Platform,
  StyleSheet,
  Text,
  View,
} from "react-native";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-react-native";
import { useEffect } from "react";
import { Camera, requestPermissionsAsync } from "expo-camera";
import { cameraWithTensors } from "@tensorflow/tfjs-react-native";
import { ExpoWebGLRenderingContext } from "expo-gl";
import * as mobilenet from "@tensorflow-models/mobilenet";
import * as Speech from "expo-speech";

const TensorCamera = cameraWithTensors(Camera);

export default function App() {
  //
  const [tfReady, setTFReady] = useState(false);
  const [cameraAllowed, setCameraAllowed] = useState(false);

  const cameraRef = useRef<Camera | null>(null);
  const modelRef = useRef<mobilenet.MobileNet | null>(null);

  useEffect(() => {
    const setup = async () => {
      await tf.ready();
      setTFReady(true);
      const response = await requestPermissionsAsync();
      if (response.granted) {
        setCameraAllowed(true);
      } else {
        alert("Camera permission is required to use this app.");
      }
      modelRef.current = await mobilenet.load({ version: 2, alpha: 1 });
    };
    setup();
  }, []);

  const [prediction, setPrediction] = useState<{
    className: string;
    probability: number;
  } | null>(null);

  useEffect(() => {
    if (prediction) {
      Speech.speak(prediction.className.split(",")[0]);
    }
  }, [prediction]);

  const handleCameraStream = async (
    images: IterableIterator<tf.Tensor3D>,
    updateCameraPreview: () => void,
    gl: ExpoWebGLRenderingContext,
    cameraTexture: WebGLTexture
  ) => {
    const loop = async () => {
      const nextImageTensor = images.next().value;
      if (modelRef.current && nextImageTensor) {
        const predictions = await modelRef.current.classify(nextImageTensor);
        if (predictions && predictions.length > 0) {
          const prediction = predictions[0];
          setPrediction(prediction);
        }
      }
      setTimeout(() => loop(), 3000);
    };
    loop();
  };

  let textureDims;
  if (Platform.OS === "ios") {
    textureDims = {
      height: 1920,
      width: 1080,
    };
  } else {
    textureDims = {
      height: 1200,
      width: 1600,
    };
  }

  return (
    <View style={styles.container}>
      {tfReady && cameraAllowed ? (
        <>
          <View style={styles.cameraContainer}>
            <TensorCamera
              // Standard Camera props
              style={[
                styles.camera,
                Platform.OS === "android"
                  ? { aspectRatio: 9 / 11 }
                  : {
                      position: "absolute",
                      height: Dimensions.get("window").height,
                      width: Dimensions.get("window").width,
                    },
              ]}
              type={Camera.Constants.Type.back}
              // Tensor related props
              cameraTextureHeight={textureDims.height}
              cameraTextureWidth={textureDims.width}
              resizeHeight={224}
              resizeWidth={224}
              resizeDepth={3}
              onReady={handleCameraStream}
              autorender={true}
              useCustomShadersToResize
              ref={(ref) =>
                ref?.camera ? (cameraRef.current = ref.camera) : null
              }
            />
          </View>
          <View style={{ marginVertical: 50 }}>
            <Text style={styles.title}>
              {prediction ? `"${prediction.className.split(",")[0]}"` : ""}
            </Text>
            <Text style={styles.confidence}>
              {prediction
                ? `Confidence: ${(prediction.probability * 100).toFixed(1)} %`
                : ""}
            </Text>
          </View>
        </>
      ) : (
        <ActivityIndicator size="large" color="#ABC" />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    height: "100%",
    width: "100%",
    backgroundColor: "#fff",
    alignItems: "center",
    justifyContent: "space-around",
    paddingTop: 50,
  },
  cameraContainer: {
    flex: 1,
    width: "90%",
    borderRadius: 20,
    overflow: "hidden",
  },
  camera: {
    flex: 1,
    width: "100%",
  },
  title: {
    fontSize: 28,
    textAlign: "center",
  },
  confidence: {
    fontSize: 14,
    textAlign: "center",
    color: "#AAA",
  },
});
