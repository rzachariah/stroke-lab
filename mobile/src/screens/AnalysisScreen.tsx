import React from "react";
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  SafeAreaView,
  TouchableOpacity,
} from "react-native";
import type { NativeStackScreenProps } from "@react-navigation/native-stack";
import type { RootStackParamList } from "../../App";

type Props = NativeStackScreenProps<RootStackParamList, "Analysis">;

function ScoreBar({ label, score }: { label: string; score: number }) {
  const pct = (score / 10) * 100;
  const color = score >= 7 ? "#4aff8c" : score >= 4 ? "#e8ff4a" : "#ff4a4a";
  return (
    <View style={bar.row}>
      <Text style={bar.label}>{label}</Text>
      <View style={bar.track}>
        <View style={[bar.fill, { width: `${pct}%`, backgroundColor: color }]} />
      </View>
      <Text style={[bar.score, { color }]}>{score.toFixed(1)}</Text>
    </View>
  );
}

export default function AnalysisScreen({ navigation, route }: Props) {
  const { result } = route.params;
  const { scores, metrics, feedback } = result;

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scroll}>
        <View style={styles.overallRow}>
          <Text style={styles.overallLabel}>Overall</Text>
          <Text style={styles.overallScore}>{scores.overall.toFixed(1)}</Text>
          <Text style={styles.overallMax}>/10</Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Power Sources</Text>
          <ScoreBar label="Legs" score={scores.legs} />
          <ScoreBar label="Shoulders / X-factor" score={scores.shoulders} />
          <ScoreBar label="Late Hit" score={scores.late_hit} />
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Metrics</Text>
          <Row label="X-factor" value={metrics.peak_x_factor != null ? `${metrics.peak_x_factor}°` : "—"} />
          <Row label="Min knee bend" value={metrics.min_knee_bend != null ? `${metrics.min_knee_bend}°` : "—"} />
          <Row label="Contact position" value={metrics.contact_wrist_x_rel != null ? (metrics.contact_wrist_x_rel > 0 ? "In front ✓" : "Behind ✗") : "—"} />
          <Row label="Swing path" value={metrics.swing_direction ?? "—"} />
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Coaching Feedback</Text>
          <Text style={styles.feedback}>{feedback}</Text>
        </View>

        <TouchableOpacity style={styles.back} onPress={() => navigation.popToTop()}>
          <Text style={styles.backText}>Analyze Another</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <View style={row.container}>
      <Text style={row.label}>{label}</Text>
      <Text style={row.value}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0a0a0a" },
  scroll: { padding: 24, gap: 16 },
  overallRow: {
    flexDirection: "row",
    alignItems: "baseline",
    marginBottom: 8,
  },
  overallLabel: { color: "#888", fontSize: 16, flex: 1 },
  overallScore: { color: "#e8ff4a", fontSize: 56, fontWeight: "800" },
  overallMax: { color: "#555", fontSize: 24, marginLeft: 4 },
  card: {
    backgroundColor: "#111",
    borderRadius: 16,
    padding: 20,
    gap: 14,
  },
  cardTitle: { color: "#fff", fontSize: 16, fontWeight: "700", marginBottom: 4 },
  feedback: { color: "#ccc", fontSize: 15, lineHeight: 24 },
  back: {
    backgroundColor: "#1c1c1c",
    borderRadius: 14,
    padding: 18,
    alignItems: "center",
    marginTop: 8,
  },
  backText: { color: "#e8ff4a", fontSize: 15, fontWeight: "700" },
});

const bar = StyleSheet.create({
  row: { flexDirection: "row", alignItems: "center", gap: 10 },
  label: { color: "#888", fontSize: 13, width: 120 },
  track: {
    flex: 1,
    height: 8,
    backgroundColor: "#222",
    borderRadius: 4,
    overflow: "hidden",
  },
  fill: { height: "100%", borderRadius: 4 },
  score: { fontSize: 14, fontWeight: "700", width: 32, textAlign: "right" },
});

const row = StyleSheet.create({
  container: { flexDirection: "row", justifyContent: "space-between" },
  label: { color: "#888", fontSize: 14 },
  value: { color: "#fff", fontSize: 14, fontWeight: "600" },
});
