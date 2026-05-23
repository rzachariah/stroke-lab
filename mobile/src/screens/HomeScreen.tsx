import React from "react";
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  SafeAreaView,
} from "react-native";
import type { NativeStackScreenProps } from "@react-navigation/native-stack";
import type { RootStackParamList } from "../../App";

type Props = NativeStackScreenProps<RootStackParamList, "Home">;

export default function HomeScreen({ navigation }: Props) {
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.logo}>StrokeLab</Text>
        <Text style={styles.tagline}>OTI stroke analysis</Text>
      </View>

      <View style={styles.buttons}>
        <TouchableOpacity
          style={[styles.btn, styles.primary]}
          onPress={() => navigation.navigate("Upload")}
        >
          <Text style={styles.btnText}>Upload Video</Text>
          <Text style={styles.btnSub}>From your library</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.btn, styles.secondary]}
          onPress={() => navigation.navigate("Record")}
        >
          <Text style={styles.btnText}>Record Stroke</Text>
          <Text style={styles.btnSub}>Use camera</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.footer}>
        <Text style={styles.footerText}>Three power sources: Legs · Shoulders · Late hit</Text>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0a0a0a" },
  header: { flex: 1, justifyContent: "center", alignItems: "center" },
  logo: { fontSize: 42, fontWeight: "800", color: "#e8ff4a", letterSpacing: -1 },
  tagline: { fontSize: 14, color: "#888", marginTop: 6, letterSpacing: 1 },
  buttons: { flex: 1, paddingHorizontal: 24, gap: 16 },
  btn: {
    borderRadius: 16,
    padding: 24,
    alignItems: "center",
  },
  primary: { backgroundColor: "#e8ff4a" },
  secondary: { backgroundColor: "#1c1c1c", borderWidth: 1, borderColor: "#333" },
  btnText: { fontSize: 18, fontWeight: "700", color: "#0a0a0a" },
  btnSub: { fontSize: 13, color: "#555", marginTop: 4 },
  footer: { paddingBottom: 32, alignItems: "center" },
  footerText: { fontSize: 12, color: "#444", letterSpacing: 0.5 },
});
