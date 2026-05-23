import React, { useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
  ActivityIndicator,
  Alert,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import type { NativeStackScreenProps } from "@react-navigation/native-stack";
import type { RootStackParamList } from "../../App";
import { analyzeVideo } from "../api/client";

type Props = NativeStackScreenProps<RootStackParamList, "Upload">;

const STROKE_TYPES = ["forehand", "backhand", "serve"] as const;

export default function UploadScreen({ navigation }: Props) {
  const [videoUri, setVideoUri] = useState<string | null>(null);
  const [strokeType, setStrokeType] = useState<"forehand" | "backhand" | "serve">("forehand");
  const [loading, setLoading] = useState(false);

  const pickVideo = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Videos,
      allowsEditing: false,
      quality: 1,
    });
    if (!result.canceled && result.assets[0]) {
      setVideoUri(result.assets[0].uri);
    }
  };

  const submit = async () => {
    if (!videoUri) return;
    setLoading(true);
    try {
      const result = await analyzeVideo(videoUri, strokeType, "r");
      navigation.navigate("Analysis", { result });
    } catch (e: unknown) {
      Alert.alert("Analysis failed", e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>Upload Stroke</Text>

      <View style={styles.section}>
        <Text style={styles.label}>Stroke Type</Text>
        <View style={styles.pills}>
          {STROKE_TYPES.map((t) => (
            <TouchableOpacity
              key={t}
              style={[styles.pill, strokeType === t && styles.pillActive]}
              onPress={() => setStrokeType(t)}
            >
              <Text style={[styles.pillText, strokeType === t && styles.pillTextActive]}>
                {t.charAt(0).toUpperCase() + t.slice(1)}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      <TouchableOpacity style={styles.picker} onPress={pickVideo}>
        {videoUri ? (
          <Text style={styles.pickerReady}>Video selected</Text>
        ) : (
          <Text style={styles.pickerPrompt}>Tap to choose video</Text>
        )}
      </TouchableOpacity>

      <TouchableOpacity
        style={[styles.analyzeBtn, !videoUri && styles.analyzeBtnDisabled]}
        onPress={submit}
        disabled={!videoUri || loading}
      >
        {loading ? (
          <ActivityIndicator color="#0a0a0a" />
        ) : (
          <Text style={styles.analyzeBtnText}>Analyze Stroke</Text>
        )}
      </TouchableOpacity>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0a0a0a", padding: 24 },
  title: { fontSize: 28, fontWeight: "800", color: "#fff", marginBottom: 32 },
  section: { marginBottom: 32 },
  label: { color: "#888", fontSize: 13, marginBottom: 12, letterSpacing: 1 },
  pills: { flexDirection: "row", gap: 10 },
  pill: {
    paddingHorizontal: 18,
    paddingVertical: 10,
    borderRadius: 100,
    borderWidth: 1,
    borderColor: "#333",
  },
  pillActive: { backgroundColor: "#e8ff4a", borderColor: "#e8ff4a" },
  pillText: { color: "#888", fontSize: 14 },
  pillTextActive: { color: "#0a0a0a", fontWeight: "700" },
  picker: {
    flex: 1,
    borderRadius: 20,
    borderWidth: 2,
    borderColor: "#222",
    borderStyle: "dashed",
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 24,
  },
  pickerPrompt: { color: "#444", fontSize: 16 },
  pickerReady: { color: "#e8ff4a", fontSize: 16, fontWeight: "700" },
  analyzeBtn: {
    backgroundColor: "#e8ff4a",
    borderRadius: 14,
    padding: 18,
    alignItems: "center",
  },
  analyzeBtnDisabled: { opacity: 0.4 },
  analyzeBtnText: { color: "#0a0a0a", fontSize: 16, fontWeight: "800" },
});
