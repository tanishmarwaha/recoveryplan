import tkinter as tk
from tkinter import messagebox, scrolledtext
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

data = [
    # Lower Back
    ['mild', 3, 'dull', 'lower back', 'active', 1, 0, 1, 'plan_lower_back_gentle'],
    ['moderate', 10, 'radiating', 'lower back', 'sedentary', 1, 1, 0, 'plan_lower_back_strengthen'],
    ['severe', 2, 'sharp', 'lower back', 'active', 0, 0, 0, 'plan_rest_doctor'],
    
    # Hip
    ['mild', 21, 'dull', 'hip', 'sedentary', 1, 0, 1, 'plan_hip_gentle'],
    ['moderate', 5, 'sharp', 'hip', 'active', 1, 0, 1, 'plan_hip_strengthen'],
    ['severe', 12, 'radiating', 'hip', 'sedentary', 0, 1, 0, 'plan_rest_doctor'],
    
    # Leg
    ['mild', 7, 'dull', 'leg', 'active', 1, 0, 1, 'plan_leg_gentle'],
    ['moderate', 14, 'radiating', 'leg', 'sedentary', 1, 0, 1, 'plan_leg_strengthen'],
    ['severe', 3, 'sharp', 'leg', 'active', 0, 0, 0, 'plan_rest_doctor'],
    
    # Shoulder
    ['mild', 6, 'dull', 'shoulder', 'active', 1, 0, 1, 'plan_shoulder_gentle'],
    ['moderate', 12, 'radiating', 'shoulder', 'sedentary', 1, 0, 1, 'plan_shoulder_strengthen'],
    ['severe', 1, 'sharp', 'shoulder', 'active', 0, 0, 0, 'plan_rest_doctor'],
    
    # Arm (new injury type)
    ['mild', 7, 'dull', 'arm', 'active', 1, 0, 1, 'plan_arm_gentle'],
    ['moderate', 14, 'radiating', 'arm', 'sedentary', 1, 0, 1, 'plan_arm_strengthen'],
    ['severe', 3, 'sharp', 'arm', 'active', 0, 0, 0, 'plan_rest_doctor'],
]

categorical_columns = [0, 2, 3, 4]
encoders = [LabelEncoder() for _ in categorical_columns]

X_raw = [row[:-1] for row in data]
y = [row[-1] for row in data]

for i, col_idx in enumerate(categorical_columns):
    encoders[i].fit([row[col_idx] for row in X_raw])

X_encoded = []
for row in X_raw:
    new_row = row[:]
    for i, col_idx in enumerate(categorical_columns):
        new_row[col_idx] = encoders[i].transform([row[col_idx]])[0]
    X_encoded.append(new_row)

X = np.array(X_encoded)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

plans = {
    'plan_rest_doctor': (
        "🛑 Rest & See Medical Professional\n"
        "- Avoid all strenuous activities\n"
        "- Ice and monitor symptoms closely\n"
        "- Immediate consultation advised\n"
    ),
    'plan_lower_back_gentle': (
        "🧘 Lower Back Gentle Recovery (7 days)\n"
        "- Cat-Cow stretch 3x10 reps\n"
        "- Pelvic Tilts 3x10 reps\n"
        "- Child’s Pose hold 30s × 3\n"
        "- Gradually add Knee-to-Chest & Hamstring stretches\n\n"
        "🏋️ Reintroduction:\n"
        "- Week 2: Push-ups 5x3 sets\n"
        "- Week 3: Light Deadlifts and Squats, low weight\n"
    ),
    'plan_lower_back_strengthen': (
        "💪 Lower Back Strengthening\n"
        "- Deadlifts (light) 3x8\n"
        "- Bird-Dog 3x10 per side\n"
        "- Planks 3x20 seconds\n"
        "- Gradually increase weights and sets\n"
    ),
    'plan_hip_gentle': (
        "🧘 Hip Mobility and Gentle Stretch\n"
        "- Hip Flexor Stretch 3x30s\n"
        "- Figure-4 Stretch 3 reps each side\n"
        "- Avoid deep squats week 1\n\n"
        "🏋️ Reintroduction:\n"
        "- Week 2: Light Deadlifts 2x10 reps\n"
        "- Week 3: Add Step-ups and Glute Bridges\n"
    ),
    'plan_hip_strengthen': (
        "💪 Hip Strengthening Plan\n"
        "- Glute Bridges 3x10\n"
        "- Step-ups 2x10 each leg\n"
        "- Bodyweight Squats 2x10\n"
        "- Add dumbbells as tolerated\n"
    ),
    'plan_leg_gentle': (
        "🧘 Gentle Leg Recovery\n"
        "- Hamstring Stretch hold 30s × 2\n"
        "- Quad Stretch 3x15s each leg\n"
        "- Calf Raises 2x10 reps\n\n"
        "🏋️ Reintroduction:\n"
        "- Week 2: Bodyweight Squats\n"
        "- Week 3: Add Goblet Squats, Push-ups\n"
    ),
    'plan_leg_strengthen': (
        "💪 Leg Strengthening\n"
        "- Squats 3x8 (light)\n"
        "- Calf Raises 3x15\n"
        "- Lunges 2x10 each leg\n"
        "- Gradual load increase\n"
    ),
    'plan_shoulder_gentle': (
        "🧘 Shoulder Stretching & Mobility\n"
        "- Cross-body Shoulder Stretch 30s each side\n"
        "- Wall Slides 2x10\n"
        "- Avoid overhead pressing for 1 week\n\n"
        "🏋️ Reintroduction:\n"
        "- Week 2: Incline Push-ups 3x8\n"
        "- Week 3: Light Shoulder Press 2x10\n"
    ),
    'plan_shoulder_strengthen': (
        "💪 Shoulder Strengthening\n"
        "- Push-ups 10x3 sets\n"
        "- Bicep Curls 12x2\n"   
        "- Shoulder Press 10x2 (light)\n"
        "- Add Dumbbell Rows Week 2\n"
    ),
    'plan_arm_gentle': (
        "🧘 Arm Recovery & Mobility (7 days)\n"
        "- Wrist Circles 3x15\n"
        "- Gentle Arm Raises 3x10\n"
        "- Avoid heavy lifting first week\n\n"
        "🏋️ Reintroduction:\n"
        "- Week 2: Bicep Curls 5x2 sets (light weight)\n"
        "- Week 3: Add Shoulder Press 3x8\n"
    ),
    'plan_arm_strengthen': (
        "💪 Arm Strengthening\n"
        "- Bicep Curls 12x3 sets\n"
        "- Shoulder Press 10x3 sets\n"
        "- Push-ups 10x3 sets\n"
        "- Gradual weight increase\n"
    ),
}

class RecoveryApp:
    def __init__(self, master):
        self.master = master
        master.title("AI Recovery Plan Assistant")

        self.fields = {
            "Pain Severity": ["mild", "moderate", "severe"],
            "Pain Duration (days)": [],
            "Pain Type": ["dull", "sharp", "radiating"],
            "Pain Location": ["lower back", "hip", "leg", "shoulder", "arm"],  
            "Activity Level": ["active", "sedentary"],
            "Can Walk Comfortably?": ["yes", "no"],
            "Previous Injury History?": ["yes", "no"],
            "Symptoms Improving?": ["yes", "no"]
        }

        self.inputs = {}
        row = 0
        for label, options in self.fields.items():
            tk.Label(master, text=label).grid(row=row, column=0, sticky='w')
            if options:
                var = tk.StringVar(value=options[0])
                dropdown = tk.OptionMenu(master, var, *options)
                dropdown.grid(row=row, column=1)
                self.inputs[label] = var
            else:
                entry = tk.Entry(master)
                entry.grid(row=row, column=1)
                self.inputs[label] = entry
            row += 1

        self.predict_button = tk.Button(master, text="Generate Plan", command=self.predict)
        self.predict_button.grid(row=row, column=0, columnspan=2, pady=10)

        self.output = scrolledtext.ScrolledText(master, width=60, height=20)
        self.output.grid(row=row + 1, column=0, columnspan=2)

    def predict(self):
        try:
            severity = self.inputs["Pain Severity"].get()
            duration = int(self.inputs["Pain Duration (days)"].get())
            pain_type = self.inputs["Pain Type"].get()
            location = self.inputs["Pain Location"].get()
            activity = self.inputs["Activity Level"].get()
            can_walk = 1 if self.inputs["Can Walk Comfortably?"].get() == "yes" else 0
            history = 1 if self.inputs["Previous Injury History?"].get() == "yes" else 0
            improving = 1 if self.inputs["Symptoms Improving?"].get() == "yes" else 0

            raw_input = [severity, duration, pain_type, location, activity, can_walk, history, improving]
            encoded_input = raw_input[:]
            for i, col_idx in enumerate(categorical_columns):
                encoded_input[col_idx] = encoders[i].transform([raw_input[col_idx]])[0]

            prediction = clf.predict([encoded_input])[0]

            # Backup check to ensure injury location matches plan
            if location not in prediction:
                location_plans = [k for k in plans.keys() if location.replace(" ", "_") in k]
                if location_plans:
                    prediction = location_plans[0]

            plan = plans.get(prediction, "No recovery plan available for your inputs.")

            self.output.delete("1.0", tk.END)
            self.output.insert(tk.END, plan)

        except Exception as e:
            messagebox.showerror("Input Error", f"Please check your inputs.\n\nError: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = RecoveryApp(root)
    root.mainloop()
