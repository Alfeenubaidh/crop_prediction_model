const API_URL = "http://localhost:8000";

export const getPrediction = async (inputData: {
  temperature: number;
  humidity: number;
  ph: number;
  rainfall: number;
}) => {
  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(inputData)
    });

    if (!response.ok) {
      throw new Error("API request failed");
    }

    return await response.json();
  } catch (error) {
    console.error("Prediction error:", error);
    throw error;
  }
};