import customtkinter as ctk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from collections import defaultdict
from tkinter import messagebox
from datetime import datetime, timedelta

# === Load and preprocess ===
data = pd.read_csv("cleaned_weather.csv")
data.dropna(inplace=True)
data['weather_label'] = data['weather_label'].str.strip().str.title()

# Discretization
def categorize_temp(t):
    if t < 5: return "Low"
    elif t <= 20: return "Medium"
    else: return "High"

def categorize_humidity(h):
    if h < 40: return "Low"
    elif h <= 70: return "Medium"
    else: return "High"

def categorize_wind(w):
    if w <= 3: return "Low"
    elif w <= 7: return "Medium"
    else: return "High"

data['temp_cat'] = data['temperature'].apply(categorize_temp)
data['humidity_cat'] = data['humidity'].apply(categorize_humidity)
data['wind_cat'] = data['wind'].apply(categorize_wind)

# Train Bayesian Network
def train_bayesian_network(df):
    model = DiscreteBayesianNetwork([
        ('temp_cat', 'weather_label'),
        ('humidity_cat', 'weather_label'),
        ('wind_cat', 'weather_label')
    ])
    model.fit(df, estimator=MaximumLikelihoodEstimator)
    infer = VariableElimination(model)
    return model, infer

bn_model, bn_infer = train_bayesian_network(data)

# === Markov Model Setup ===
states = sorted(data['weather_label'].unique())
icons = {
    "Sunny": "☀", "Rainy": "🌧", "Thunderstorm": "⚡", "Cloudy": "☁", "Fog": "🌫",
    "Snow": "❄", "Mist": "🌫", "Clear": "☀", "Strong Wind": "💨", "Partly Cloudy": "⛅", "Storm": "⚡"
}
state_to_index = {state: i for i, state in enumerate(states)}
index_to_state = {i: state for state, i in state_to_index.items()}

def build_transition_matrix(sequence):
    transition_counts = defaultdict(lambda: defaultdict(int))
    for current, next_ in zip(sequence[:-1], sequence[1:]):
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

# === Continuous Prediction ===
def estimate_continuous_values_from_categories(temp_cat, humidity_cat, wind_cat, state):
    # First try exact match
    exact_subset = data[
        (data['temp_cat'] == temp_cat) &
        (data['humidity_cat'] == humidity_cat) &
        (data['wind_cat'] == wind_cat) &
        (data['weather_label'] == state)
    ]
    
    if len(exact_subset) > 0:
        return {
            "temperature": exact_subset['temperature'].mean().round(1),
            "humidity": exact_subset['humidity'].mean().round(1),
            "wind": exact_subset['wind'].mean().round(1)
        }
    
    # If no exact match, try partial matches with weighted averages
    partial_subsets = []
    weights = []
    
    # Try matching 2 out of 3 categories + weather state
    for cols in [['temp_cat', 'humidity_cat'], 
                 ['temp_cat', 'wind_cat'], 
                 ['humidity_cat', 'wind_cat']]:
        query = (data['weather_label'] == state)
        for col in cols:
            if col == 'temp_cat':
                query &= (data['temp_cat'] == temp_cat)
            elif col == 'humidity_cat':
                query &= (data['humidity_cat'] == humidity_cat)
            elif col == 'wind_cat':
                query &= (data['wind_cat'] == wind_cat)
        
        subset = data[query]
        if len(subset) > 0:
            partial_subsets.append(subset)
            weights.append(2)  # Higher weight for 2/3 matches
    
    # Try matching just 1 category + weather state
    for col in ['temp_cat', 'humidity_cat', 'wind_cat']:
        query = (data['weather_label'] == state)
        if col == 'temp_cat':
            query &= (data['temp_cat'] == temp_cat)
        elif col == 'humidity_cat':
            query &= (data['humidity_cat'] == humidity_cat)
        elif col == 'wind_cat':
            query &= (data['wind_cat'] == wind_cat)
        
        subset = data[query]
        if len(subset) > 0:
            partial_subsets.append(subset)
            weights.append(1)  # Lower weight for 1/3 matches
    
    # If we found any partial matches
    if len(partial_subsets) > 0:
        # Calculate weighted averages
        total_weight = sum(weights)
        temp_sum = hum_sum = wind_sum = 0
        
        for subset, weight in zip(partial_subsets, weights):
            temp_sum += subset['temperature'].mean() * weight
            hum_sum += subset['humidity'].mean() * weight
            wind_sum += subset['wind'].mean() * weight
        
        return {
            "temperature": round(temp_sum / total_weight, 1),
            "humidity": round(hum_sum / total_weight, 1),
            "wind": round(wind_sum / total_weight, 1)
        }
    
    # Final fallback - just use averages for the weather state
    state_subset = data[data['weather_label'] == state]
    if len(state_subset) > 0:
        return {
            "temperature": state_subset['temperature'].mean().round(1),
            "humidity": state_subset['humidity'].mean().round(1),
            "wind": state_subset['wind'].mean().round(1)
        }
    
    # Ultimate fallback - use overall averages
    return {
        "temperature": data['temperature'].mean().round(1),
        "humidity": data['humidity'].mean().round(1),
        "wind": data['wind'].mean().round(1)
    }

def predict_weather_bayes(temp, humidity, wind):
    evidence = {
        'temp_cat': categorize_temp(temp),
        'humidity_cat': categorize_humidity(humidity),
        'wind_cat': categorize_wind(wind)
    }
    prediction = bn_infer.map_query(variables=['weather_label'], evidence=evidence)
    return prediction['weather_label']

# === Visualization ===
def show_transition_graph(current_state):
    G = nx.DiGraph()
    probs = transition_matrix[state_to_index[current_state]]

    for i, prob in enumerate(probs):
        if prob > 0:
            G.add_edge(current_state, index_to_state[i], weight=round(prob, 2))

    pos = nx.spring_layout(G, seed=42)
    edge_labels = nx.get_edge_attributes(G, 'weight')

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title(f"Transition Probabilities from '{current_state}'")
    plt.show()

# === GUI ===
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class WeatherApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Weather Predictor")
        self.geometry("1000x750")
        self.current_unit = "C"
        self.current_prediction = None
        self.forecast_data = [None] * 7
        self.build_ui()
        self.set_default_values()
        self.update_forecast_dates()

    def build_ui(self):
        self.grid_columnconfigure(0, weight=1)

        # Header
        self.header_frame = ctk.CTkFrame(self)
        self.header_frame.pack(pady=(20, 10), padx=20, fill="x")
        
        self.location_label = ctk.CTkLabel(self.header_frame, text="3mman Weather", font=("Segoe UI", 24, "bold"))
        self.location_label.pack(side="left", padx=10)
        
        self.theme_switch = ctk.CTkSwitch(self.header_frame, text="Light Mode", command=self.toggle_theme)
        self.theme_switch.pack(side="right", padx=10)

        # Current weather display
        self.current_weather_frame = ctk.CTkFrame(self)
        self.current_weather_frame.pack(pady=10, padx=20, fill="x")

        self.current_weather_icon = ctk.CTkLabel(self.current_weather_frame, text="☀", font=("Segoe UI", 40))
        self.current_weather_icon.grid(row=0, column=0, padx=10, rowspan=2)

        self.current_weather_text = ctk.CTkLabel(self.current_weather_frame, text="Enter values and click Predict", 
                                               font=("Segoe UI", 16))
        self.current_weather_text.grid(row=0, column=1, padx=10, sticky="w")

        self.current_temp_label = ctk.CTkLabel(self.current_weather_frame, text="--", font=("Segoe UI", 16))
        self.current_temp_label.grid(row=0, column=2, padx=10, sticky="w")

        self.current_humidity_label = ctk.CTkLabel(self.current_weather_frame, text="Humidity: --%", 
                                                  font=("Segoe UI", 12))
        self.current_humidity_label.grid(row=1, column=1, padx=10, sticky="w")

        self.current_wind_label = ctk.CTkLabel(self.current_weather_frame, text="Wind: -- m/h", 
                                             font=("Segoe UI", 12))
        self.current_wind_label.grid(row=1, column=2, padx=10, sticky="w")

        # Input section
        input_frame = ctk.CTkFrame(self)
        input_frame.pack(pady=20, padx=20, fill="x")

        ctk.CTkLabel(input_frame, text="Temperature (°C):").grid(row=0, column=0, padx=5, sticky="w")
        self.temp_input = ctk.CTkEntry(input_frame, placeholder_text="e.g. 15")
        self.temp_input.grid(row=1, column=0, padx=5)

        ctk.CTkLabel(input_frame, text="Humidity (%):").grid(row=0, column=1, padx=5, sticky="w")
        self.humidity_input = ctk.CTkEntry(input_frame, placeholder_text="e.g. 65")
        self.humidity_input.grid(row=1, column=1, padx=5)

        ctk.CTkLabel(input_frame, text="Wind Speed (m/h):").grid(row=0, column=2, padx=5, sticky="w")
        self.wind_input = ctk.CTkEntry(input_frame, placeholder_text="e.g. 12")
        self.wind_input.grid(row=1, column=2, padx=5)

        ctk.CTkLabel(input_frame, text="Days to Forecast:").grid(row=0, column=3, padx=5, sticky="w")
        self.days_input = ctk.CTkSlider(input_frame, from_=1, to=7, number_of_steps=6)
        self.days_input.set(3)
        self.days_input.grid(row=1, column=3, padx=10, sticky="ew")

        # Input value display
        self.input_display_frame = ctk.CTkFrame(self)
        self.input_display_frame.pack(pady=10, padx=20, fill="x")

        self.input_temp_label = ctk.CTkLabel(self.input_display_frame, text="Temperature Entered: --°C")
        self.input_temp_label.pack(side="left", padx=10)

        self.input_humidity_label = ctk.CTkLabel(self.input_display_frame, text="Humidity Entered: --%")
        self.input_humidity_label.pack(side="left", padx=10)

        self.input_wind_label = ctk.CTkLabel(self.input_display_frame, text="Wind Speed Entered: -- m/h")
        self.input_wind_label.pack(side="left", padx=10)

        # Button section
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(pady=10)

        self.predict_btn = ctk.CTkButton(button_frame, text="Predict Weather", command=self.start_prediction)
        self.predict_btn.pack(side="left", padx=5)

        self.graph_btn = ctk.CTkButton(button_frame, text="Show Graph", command=self.show_graph, state="disabled")
        self.graph_btn.pack(side="left", padx=5)

        # Unit selection
        self.unit_frame = ctk.CTkFrame(self)
        self.unit_frame.pack(pady=5)
        self.unit_var = ctk.StringVar(value="C")
        ctk.CTkLabel(self.unit_frame, text="Temperature Unit:").pack(side="left", padx=5)
        ctk.CTkRadioButton(self.unit_frame, text="°C", variable=self.unit_var, value="C", command=self.update_units).pack(side="left", padx=5)
        ctk.CTkRadioButton(self.unit_frame, text="°F", variable=self.unit_var, value="F", command=self.update_units).pack(side="left", padx=5)
        ctk.CTkRadioButton(self.unit_frame, text="K", variable=self.unit_var, value="K", command=self.update_units).pack(side="left", padx=5)

        # Forecast display
        self.forecast_frame = ctk.CTkFrame(self)
        self.forecast_frame.pack(pady=10, padx=20, fill="both", expand=True)

        # Create forecast day columns
        self.day_frames = []
        self.day_labels = []
        self.date_labels = []
        self.weather_icons = []
        self.temp_labels = []
        self.humidity_labels = []
        self.wind_labels = []

        for i in range(7):
            frame = ctk.CTkFrame(self.forecast_frame)
            frame.grid(row=0, column=i, padx=5, pady=5, sticky="nsew")
            self.forecast_frame.columnconfigure(i, weight=1)
            self.day_frames.append(frame)

            day_label = ctk.CTkLabel(frame, text=f"Day {i+1}", font=("Segoe UI", 12, "bold"))
            day_label.pack(pady=(5, 0))
            self.day_labels.append(day_label)

            date_label = ctk.CTkLabel(frame, text="", font=("Segoe UI", 10))
            date_label.pack()
            self.date_labels.append(date_label)

            icon = ctk.CTkLabel(frame, text="", font=("Segoe UI", 24))
            icon.pack(pady=5)
            self.weather_icons.append(icon)

            temp_label = ctk.CTkLabel(frame, text="--", font=("Segoe UI", 14))
            temp_label.pack()
            self.temp_labels.append(temp_label)

            humidity_label = ctk.CTkLabel(frame, text="Humidity: --%", font=("Segoe UI", 10))
            humidity_label.pack()
            self.humidity_labels.append(humidity_label)

            wind_label = ctk.CTkLabel(frame, text="Wind: -- m/h", font=("Segoe UI", 10))
            wind_label.pack()
            self.wind_labels.append(wind_label)

    def set_default_values(self):
        self.temp_input.insert(0, "15")
        self.humidity_input.insert(0, "65")
        self.wind_input.insert(0, "12")

    def update_forecast_dates(self):
        today = datetime.now()
        for i in range(7):
            forecast_date = today + timedelta(days=i+1)
            self.date_labels[i].configure(text=forecast_date.strftime("%b %d"))

    def toggle_theme(self):
        if self.theme_switch.get() == 1:
            ctk.set_appearance_mode("light")
            self.theme_switch.configure(text="Dark Mode")
        else:
            ctk.set_appearance_mode("dark")
            self.theme_switch.configure(text="Light Mode")

    def convert_temp(self, temp_c, unit):
        if unit == "F":
            return (temp_c * 9 / 5) + 32, "°F"
        elif unit == "K":
            return temp_c + 273.15, "K"
        return temp_c, "°C"

    def update_units(self):
        self.current_unit = self.unit_var.get()
        self.update_current_weather_display()
        self.update_forecast_display()

    def update_current_weather_display(self):
        if self.current_prediction:
            temp, unit = self.convert_temp(self.current_prediction["temperature"], self.current_unit)
            self.current_temp_label.configure(text=f"{temp:.1f}{unit}")
            self.current_humidity_label.configure(text=f"Humidity: {self.current_prediction['humidity']}%")
            self.current_wind_label.configure(text=f"Wind: {self.current_prediction['wind']} m/h")

    def update_forecast_display(self):
        for i in range(7):
            if self.forecast_data[i] is not None:
                temp, unit = self.convert_temp(self.forecast_data[i]["temperature"], self.current_unit)
                self.temp_labels[i].configure(text=f"{temp:.1f}{unit}")
                self.humidity_labels[i].configure(text=f"Humidity: {self.forecast_data[i]['humidity']}%")
                self.wind_labels[i].configure(text=f"Wind: {self.forecast_data[i]['wind']} m/h")

    def validate_inputs(self):
        try:
            temp = float(self.temp_input.get())
            humidity = float(self.humidity_input.get())
            wind = float(self.wind_input.get())
            
            if not (0 <= humidity <= 100):
                raise ValueError("Humidity must be between 0-100%")
            if wind < 0:
                raise ValueError("Wind speed cannot be negative")
            
            return True
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return False

    def start_prediction(self):
        if not self.validate_inputs():
            return
            
        self.predict_btn.configure(state="disabled", text="Predicting...")
        self.graph_btn.configure(state="disabled")
        self.after(100, self.run_prediction)

    def run_prediction(self):
        try:
            # Get user inputs
            temp = float(self.temp_input.get())
            humidity = float(self.humidity_input.get())
            wind = float(self.wind_input.get())
            days = int(self.days_input.get())

            # Update input labels
            self.input_temp_label.configure(text=f"Temperature Entered: {temp:.1f}°C")
            self.input_humidity_label.configure(text=f"Humidity Entered: {humidity:.1f}%")
            self.input_wind_label.configure(text=f"Wind Speed Entered: {wind:.1f} m/h")

            # Bayesian prediction for today's weather state
            predicted_weather = predict_weather_bayes(temp, humidity, wind)

            # Store today's weather using USER INPUT VALUES
            self.current_prediction = {
                "temperature": temp,  # Use the input temperature directly
                "humidity": humidity,  # Use the input humidity directly
                "wind": wind,  # Use the input wind speed directly
                "weather_state": predicted_weather  # Store the predicted state
            }

            # Display today's weather (using input values)
            self.current_weather_text.configure(text=f"Today's Weather: {predicted_weather}")
            self.current_weather_icon.configure(text=icons.get(predicted_weather, "☀"))
            self.update_current_weather_display()

            # Forecast prediction using Markov Model (using predicted values for future days)
            markov_predictions = predict_weather_markov(predicted_weather, days, transition_matrix)
            self.forecast_data = [None] * 7

            for i, weather in enumerate(markov_predictions):
                forecast = estimate_continuous_values_from_categories(
                    categorize_temp(temp),
                    categorize_humidity(humidity),
                    categorize_wind(wind),
                    weather
                )

                if forecast:
                    self.forecast_data[i] = forecast
                    self.weather_icons[i].configure(text=icons.get(weather, "☀"))
                    self.day_labels[i].configure(text=f"Day {i+1} ({weather})")

            self.update_forecast_display()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while predicting: {str(e)}")
        finally:
            self.predict_btn.configure(state="normal", text="Predict Weather")
            self.graph_btn.configure(state="normal")

    def show_graph(self):
        if self.current_prediction:
            current_weather = self.current_weather_text.cget("text").replace("Today's Weather: ", "")
            show_transition_graph(current_weather)

if __name__ == "__main__":
    app = WeatherApp()
    app.mainloop()