from classtwin.twin.classroom_env import ClassroomDigitalTwinEnv

def test():
    print("Testing ClassroomDigitalTwinEnv with empirical multi-source data...")
    env = ClassroomDigitalTwinEnv(use_real_data=True)
    s0 = env.reset()
    print(f"Cohort loaded: {len(env.students)} students from real classroom!")
    print(f"State shape: {s0.shape}")
    print(f"Mean Mastery: {s0[0]:.4f}, Q25: {s0[1]:.4f}, Q75: {s0[2]:.4f}")
    print(f"Environmental readings: CO2={env.context_env['co2_ppm']} ppm, Temp={env.context_env['temp_c']} C")

    actions = ["lecture", "quiz", "groupwork", "discussion", "break"]
    for act in actions:
        s_next, r, term, info = env.step(act, duration_minutes=5.0)
        print(f"Action '{act:10s}' -> Reward={r:+.4f}, Delta_Mean_M={info['delta_mean_mastery']:+.4f}, Delta_Q25={info['delta_low_mastery']:+.4f}")

    print("\nAll actions successfully stepped through multi-source grounded dynamics!")

if __name__ == "__main__":
    test()
