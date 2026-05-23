import React, { useRef, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  Alert,
  ActivityIndicator,
} from "react-native";
import { CameraView, useCameraPermissions, CameraRecordingOptions } from "expo-camera";
import type { NativeStackScreenProps } from "@react-navigation/native-stack";
import type { RootStackParamList } from "../../App";
import { analyzeVideo } from "../api/client";

type Props = NativeStackScreenProps<RootStackParamList, "Record">;

export default function RecordScreen({ navigation }: Props) {
  const [permission, requestPermission] = useCameraPermissions();
  const [recording, setRecording] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const cameraRef = useRef<CameraView>(null);

  if (!permission) return <View style={styles.container} />;

  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.container}>
        <Text style={styles.permText}>Camera access needed to record strokes.</Text>
        <TouchableOpacity style={styles.permBtn} onPress={requestPermission}>
          <Text style={styles.permBtnText}>Allow Camera</Text>
        </TouchableOpacity>
      </SafeAreaView>
    );
  }

  const startRecording = async () => {
    if (!cameraRef.current) return;
    setRecording(true);
    try {
      const opts: CameraRecordingOptions = { maxDuration: 30 };
      const video = await cameraRef.current.recordAsync(opts);
      if (video?.uri) {
        setAnalyzing(true);
        const result = await analyzeVideo(video.uri, "forehand", "r");
        navigation.navigate("Analysis", { result });
      }
    } catch (e: unknown) {
      Alert.alert("Error", e instanceof Error ? e.message : String(e));
    } finally {
      setRecording(false);
      setAnalyzing(false);
    }
  };

  const stopRecording = () => {
    cameraRef.current?.stopRecording();
  };

  if (analyzing) {
    return (
      <SafeAreaView style={styles.container}>
        <ActivityIndicator size="large" color="#e8ff4a" />
        <Text style={styles.analyzingText}>Analyzing your stroke...</Text>
      </SafeAreaView>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView ref={cameraRef} style={styles.camera} facing="back" mode="video">
        <SafeAreaView style={styles.overlay}>
          <Text style={styles.hint}>
            {recording ? "Recording... tap to stop" : "Position camera behind player"}
          </Text>
          <TouchableOpacity
            style={[styles.recordBtn, recording && styles.recordBtnActive]}
            onPress={recording ? stopRecording : startRecording}
          >
            <View style={[styles.recordInner, recording && styles.recordInnerActive]} />
          </TouchableOpacity>
        </SafeAreaView>
      </CameraView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0a0a0a", justifyContent: "center", alignItems: "center" },
  camera: { flex: 1, width: "100%" },
  overlay: { flex: 1, justifyContent: "flex-end", alignItems: "center", paddingBottom: 48 },
  hint: { color: "#fff", fontSize: 14, marginBottom: 24, opacity: 0.8 },
  recordBtn: {
    width: 72,
    height: 72,
    borderRadius: 36,
    borderWidth: 4,
    borderColor: "#fff",
    justifyContent: "center",
    alignItems: "center",
  },
  recordBtnActive: { borderColor: "#ff4a4a" },
  recordInner: { width: 52, height: 52, borderRadius: 26, backgroundColor: "#ff4a4a" },
  recordInnerActive: { width: 24, height: 24, borderRadius: 4 },
  permText: { color: "#888", fontSize: 16, textAlign: "center", margin: 24 },
  permBtn: { backgroundColor: "#e8ff4a", borderRadius: 12, padding: 16, marginHorizontal: 24 },
  permBtnText: { color: "#0a0a0a", fontWeight: "700", textAlign: "center" },
  analyzingText: { color: "#888", marginTop: 16 },
});
