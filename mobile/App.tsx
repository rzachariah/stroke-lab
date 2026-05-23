import React from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import type { AnalysisResult } from "./src/api/client";

import HomeScreen from "./src/screens/HomeScreen";
import UploadScreen from "./src/screens/UploadScreen";
import RecordScreen from "./src/screens/RecordScreen";
import AnalysisScreen from "./src/screens/AnalysisScreen";

export type RootStackParamList = {
  Home: undefined;
  Upload: undefined;
  Record: undefined;
  Analysis: { result: AnalysisResult };
};

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName="Home"
        screenOptions={{
          headerStyle: { backgroundColor: "#0a0a0a" },
          headerTintColor: "#e8ff4a",
          headerTitleStyle: { fontWeight: "800" },
          contentStyle: { backgroundColor: "#0a0a0a" },
        }}
      >
        <Stack.Screen name="Home" component={HomeScreen} options={{ headerShown: false }} />
        <Stack.Screen name="Upload" component={UploadScreen} options={{ title: "Upload Stroke" }} />
        <Stack.Screen name="Record" component={RecordScreen} options={{ title: "Record Stroke", headerTransparent: true }} />
        <Stack.Screen name="Analysis" component={AnalysisScreen} options={{ title: "Analysis" }} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
