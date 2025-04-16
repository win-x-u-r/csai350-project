import customtkinter as ctk
import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork  # If using this lib instead of pgmpy
from collections import defaultdict
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pyplot as plt

# === Load and preprocess data ===
data = pd.read_csv('cleaned_weather.csv')
data.dropna(inplace=True)
data['weather_label'] = data['weather_label'].str.strip().str.title()

# Weather states and icons
states = sorted(data['weather_label'].unique())
icons = {"Sunny": "☀️", "Rainy": "🌧️", "Thunderstorm": "🌩️", "Cloudy": "☁️", "Fog": "🌫️", "Snow": "❄️"}
state_to_index = {state: i for i, state in enumerate(states)}
index_to_state = {i: state for state, i in state_to_index.items()}

# === Estimate continuous values for a given state ===
def estimate_continuous_values(state):
    subset = data[data['weather_label'] == state]
    if len(subset) == 0:
        return {"temperature": np.nan, "humidity": np.nan, "wind": np.nan}
    return {
        "temperature": subset['temperature'].mean().round(1),
        "humidity": subset['humidity'].mean().round(1),
        "wind": subset['wind'].mean().round(1)
    }

# === Transition Matrix ===
def build_transition_matrix(weather_sequence):
    transition_counts = defaultdict(lambda: defaultdict(int))
    for current, next_ in zip(weather_sequence[:-1], weather_sequence[1:]):
        transition_counts[current][next_] += 1
    matrix = np.zeros((len(states), len(states)))
    for i, curr in enumerate(states):
        total = sum(transition_counts[curr].values())
        for j, next_ in enumerate(states):
            matrix[i, j] = transition_counts[curr].get(next_, 0) / total if total > 0 else 1 / len(states)
    return matrix

transition_matrix = build_transition_matrix(data['weather_label'].values)

def predict_weather_markov(current_state, days, matrix):
    index = state_to_index[current_state]
    predictions = []
    for _ in range(days):
        next_index = np.random.choice(len(states), p=matrix[index])
        predictions.append(index_to_state[next_index])
        index = next_index
    return predictions

# === GUI App ===
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class WeatherApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Weather Predictor")
        self.geometry("900x600")
        self.current_unit = "C"
        self.build_ui()

    def build_ui(self):
        self.grid_columnconfigure(0, weight=1)

        self.location_label = ctk.CTkLabel(self, text="3mman", font=("Segoe UI", 24, "bold"))
        self.location_label.pack(pady=(20, 5))

        self.current_weather_frame = ctk.CTkFrame(self)
        self.current_weather_frame.pack(pady=10)

        self.current_weather_icon = ctk.CTkLabel(self.current_weather_frame, text="☀️", font=("Segoe UI", 40))
        self.current_weather_icon.grid(row=0, column=0, padx=10)

        self.current_weather_text = ctk.CTkLabel(self.current_weather_frame, text="Partly cloudy", font=("Segoe UI", 16))
        self.current_weather_text.grid(row=0, column=1, padx=10)

        self.current_temp_label = ctk.CTkLabel(self.current_weather_frame, text="15°C", font=("Segoe UI", 16))
        self.current_temp_label.grid(row=0, column=2, padx=10)

        input_frame = ctk.CTkFrame(self)
        input_frame.pack(pady=20, padx=20, fill="x")

        ctk.CTkLabel(input_frame, text="Current Weather:").grid(row=0, column=0, padx=5, sticky="w")
        self.weather_input = ctk.CTkOptionMenu(input_frame, values=states, width=150)
        self.weather_input.set(states[0])
        self.weather_input.grid(row=1, column=0, padx=5, pady=5)

        ctk.CTkLabel(input_frame, text="Temperature (°C):").grid(row=0, column=1, padx=5, sticky="w")
        self.temp_input = ctk.CTkEntry(input_frame, placeholder_text="e.g. 15")
        self.temp_input.grid(row=1, column=1, padx=5)

        ctk.CTkLabel(input_frame, text="Humidity (%):").grid(row=0, column=2, padx=5, sticky="w")
        self.humidity_input = ctk.CTkEntry(input_frame, placeholder_text="e.g. 65")
        self.humidity_input.grid(row=1, column=2, padx=5)

        ctk.CTkLabel(input_frame, text="Wind Speed (km/h):").grid(row=0, column=3, padx=5, sticky="w")
        self.wind_input = ctk.CTkEntry(input_frame, placeholder_text="e.g. 12")
        self.wind_input.grid(row=1, column=3, padx=5)

        ctk.CTkLabel(input_frame, text="Days to Forecast:").grid(row=0, column=4, padx=5, sticky="w")
        self.days_input = ctk.CTkSlider(input_frame, from_=1, to=7, number_of_steps=6)
        self.days_input.set(3)
        self.days_input.grid(row=1, column=4, padx=10, sticky="ew")

        self.predict_btn = ctk.CTkButton(self, text="Predict Weather", command=self.run_prediction)
        self.predict_btn.pack(pady=10)

        self.unit_frame = ctk.CTkFrame(self)
        self.unit_frame.pack(pady=5)
        self.unit_var = ctk.StringVar(value="C")
        ctk.CTkLabel(self.unit_frame, text="Temperature Unit:").pack(side="left", padx=5)
        ctk.CTkRadioButton(self.unit_frame, text="°C", variable=self.unit_var, value="C", command=self.update_units).pack(side="left", padx=5)
        ctk.CTkRadioButton(self.unit_frame, text="°F", variable=self.unit_var, value="F", command=self.update_units).pack(side="left", padx=5)
        ctk.CTkRadioButton(self.unit_frame, text="K", variable=self.unit_var, value="K", command=self.update_units).pack(side="left", padx=5)

        self.forecast_frame = ctk.CTkFrame(self)
        self.forecast_frame.pack(pady=10, padx=20, fill="both", expand=True)

        self.day_labels = []
        self.temp_labels = []
        self.weather_icons = []

        for i in range(7):
            self.forecast_frame.grid_columnconfigure(i, weight=1)

            day = ctk.CTkLabel(self.forecast_frame, text="", font=("Segoe UI", 12))
            day.grid(row=0, column=i, padx=10, pady=5)
            self.day_labels.append(day)

            temp = ctk.CTkLabel(self.forecast_frame, text="--", font=("Segoe UI", 14, "bold"))
            temp.grid(row=1, column=i, padx=10)
            self.temp_labels.append(temp)

            icon = ctk.CTkLabel(self.forecast_frame, text="", font=("Segoe UI", 24))
            icon.grid(row=2, column=i, padx=10)
            self.weather_icons.append(icon)

    def convert_temp(self, temp_c, unit):
        if unit == "C":
            return temp_c, "°C"
        elif unit == "F":
            return (temp_c * 9 / 5) + 32, "°F"
        elif unit == "K":
            return temp_c + 273.15, "K"
        return temp_c, "°C"

    def update_units(self):
        self.current_unit = self.unit_var.get()
        self.update_current_weather_display()
        self.update_forecast_display()

    def update_current_weather_display(self):
        try:
            temp_c = float(self.temp_input.get())
            temp, unit_symbol = self.convert_temp(temp_c, self.current_unit)
            self.current_temp_label.configure(text=f"{temp:.1f}{unit_symbol}")
        except:
            self.current_temp_label.configure(text="--")

    def update_forecast_display(self):
        for i in range(7):
            current_text = self.temp_labels[i].cget("text")
            if current_text != "--":
                try:
                    temp_c = float(current_text[:-2])
                    temp, unit_symbol = self.convert_temp(temp_c, self.current_unit)
                    self.temp_labels[i].configure(text=f"{temp:.1f}{unit_symbol}")
                except:
                    pass

    def run_prediction(self):
        try:
            weather = self.weather_input.get()
            temp_c = float(self.temp_input.get())
            humidity = float(self.humidity_input.get())
            wind = float(self.wind_input.get())
            days = int(self.days_input.get())

            self.current_weather_icon.configure(text=icons.get(weather, "☀️"))
            self.current_weather_text.configure(text=weather)
            self.update_current_weather_display()

            predictions = predict_weather_markov(weather, days, transition_matrix)

            day_names = ["Today", "Tomorrow", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]

            for i in range(7):
                if i < days:
                    state = predictions[i]
                    values = estimate_continuous_values(state)
                    temp_c = values["temperature"]
                    temp, unit_symbol = self.convert_temp(temp_c, self.current_unit)

                    self.day_labels[i].configure(text=day_names[i])
                    self.temp_labels[i].configure(text=f"{temp:.1f}{unit_symbol}")
                    self.weather_icons[i].configure(text=icons.get(state, "☀️"))
                else:
                    self.day_labels[i].configure(text="")
                    self.temp_labels[i].configure(text="--")
                    self.weather_icons[i].configure(text="")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    app = WeatherApp()
    app.mainloop()
    # end t