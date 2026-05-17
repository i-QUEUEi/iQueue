"""CLI interface — the terminal menu that users interact with.

Provides 3 prediction views:
1. Weekly forecast — shows all 6 working days of a chosen week
2. Daily forecast — shows hour-by-hour breakdown for one specific date
3. Best time finder — identifies the optimal hour to visit on a given date
"""
import numpy as np
import pandas as pd

from .constants import MONTE_CARLO_RUNS
from .inference import get_congestion_level, get_holiday_name, predict_wait_time_monte_carlo
from .inference import get_holiday_flags


def display_weekly_forecast(target_date):
    """Show predictions for all 6 working days (Mon–Sat) of a given week.

    For each day, displays:
    - Overall average wait time and congestion level
    - Best time to visit (hour with shortest predicted wait)
    - Worst time to avoid (hour with longest predicted wait)
    - P10-P90 confidence ranges

    Args:
        target_date: Any date within the target week.
    """
    week_of_month = (target_date.day - 1) // 7 + 1
    # Calculate the Monday of this week (go back by weekday number of days)
    week_start = target_date - pd.Timedelta(days=target_date.weekday())
    print("\n" + "=" * 80)
    print(
        "📅 WEEKLY CONGESTION FORECAST — Week of {} (Week {} of month)".format(
            week_start.strftime("%B %d, %Y"), week_of_month
        )
    )
    print("=" * 80)

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    # Loop through each working day of the week
    for idx, day in enumerate(days):
        day_date = week_start + pd.Timedelta(days=idx)  # Calculate the actual date for this day

        # ── Holiday guard ──────────────────────────────────────────────────────
        # LTO is closed on Philippine holidays — skip congestion prediction
        is_hol, _ = get_holiday_flags(day_date)
        holiday_name = get_holiday_name(day_date) if is_hol else None
        if is_hol:
            print(f"\n{day} ({day_date.strftime('%b %d, %Y')}):")
            print("-" * 40)
            print(f"   ⛔ CLOSED — {holiday_name}")
            print("   LTO CDO does not operate on public holidays.")
            continue
        # ──────────────────────────────────────────────────────────────────────

        print(f"\n{day} ({day_date.strftime('%b %d, %Y')}):")
        print("-" * 40)

        # Run Monte Carlo predictions for each hour of the day
        hourly_waits = []
        hourly_ranges = []
        for hour in range(8, 17):  # 8am to 4pm (office hours)
            mc = predict_wait_time_monte_carlo(day_date, hour)
            hourly_waits.append(mc["mean"])
            hourly_ranges.append((mc["p10"], mc["p90"]))

        # Find the best and worst hours
        avg_wait = np.mean(hourly_waits)
        best_idx = np.argmin(hourly_waits)    # Index of shortest wait
        worst_idx = np.argmax(hourly_waits)   # Index of longest wait
        level, rec = get_congestion_level(avg_wait)

        # Display summary for this day
        print(f"   📊 Overall: {avg_wait:.0f} min average ({level})")
        print(
            "   ⏰ Best time: {:02d}:00 ({} min, P10-P90 {}-{})".format(
                best_idx + 8,
                f"{hourly_waits[best_idx]:.0f}",
                f"{hourly_ranges[best_idx][0]:.0f}",
                f"{hourly_ranges[best_idx][1]:.0f}",
            )
        )
        print(
            "   ⚠️ Worst time: {:02d}:00 ({} min, P10-P90 {}-{})".format(
                worst_idx + 8,
                f"{hourly_waits[worst_idx]:.0f}",
                f"{hourly_ranges[worst_idx][0]:.0f}",
                f"{hourly_ranges[worst_idx][1]:.0f}",
            )
        )
        print(f"   💡 {rec}")


def display_daily_forecast(target_date):
    """Show hour-by-hour predictions for a specific date.

    For each hour (8am–4pm), displays:
    - Predicted wait time with congestion level
    - P10-P90 confidence range
    - Estimated queue length
    - Visual bar chart

    Args:
        target_date: The specific date to forecast.
    """
    day_name = target_date.strftime("%A")
    week_of_month = (target_date.day - 1) // 7 + 1

    # ── Holiday guard ──────────────────────────────────────────────────────────
    # LTO is closed on Philippine holidays — show a clear message and return early
    is_hol, _ = get_holiday_flags(target_date)
    if is_hol:
        holiday_name = get_holiday_name(target_date)
        print("\n" + "=" * 80)
        print(f"⛔ LTO CDO IS CLOSED ON {day_name.upper()} {target_date.strftime('%B %d, %Y')}")
        print(f"   Reason: {holiday_name}")
        print("   Please choose a different date (Mon–Sat, non-holiday).")
        print("=" * 80)
        return
    # ──────────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 80)
    print(
        "⏰ HOURLY FORECAST FOR {} — {} (Week {} of month)".format(
            day_name.upper(), target_date.strftime("%B %d, %Y"), week_of_month
        )
    )
    print(f"📊 Based on Week-{week_of_month} {day_name} historical patterns")
    print("=" * 80)
    print(f"\n{'Time':<12} {'Wait':<12} {'Range':<14} {'Level':<12} {'Queue':<12} {'Recommendation'}")
    print("-" * 80)

    # Loop through each office hour
    for hour in range(8, 17):
        mc = predict_wait_time_monte_carlo(target_date, hour)
        wait = mc["mean"]
        level, rec = get_congestion_level(wait)

        # Build a visual bar chart (█ for wait, ░ for remaining)
        bar_length = min(20, int(wait / 4))       # Scale: 4 min per block
        bar = "█" * bar_length + "░" * (20 - bar_length)

        # Determine time of day label
        if hour < 12:
            period = "🌅 Morning"
        elif hour < 13:
            period = "🍽️ Lunch"
        else:
            period = "🌆 Afternoon"

        # Print the forecast for this hour
        print(f"\n{hour:02d}:00 ({period})")
        print(f"   Wait: {wait:.0f} minutes ({level})")
        print(f"   Likely range (P10-P90): {mc['p10']:.0f}-{mc['p90']:.0f} min")
        print(f"   Queue: ~{mc['queue_mean']:.0f} people (Week-{week_of_month} {day_name} avg)")
        print(f"   [{bar}]")
        print(f"   {rec}")


def find_best_time(target_date):
    """Identify the single best and worst hours to visit on a specific date.

    Runs predictions for all 9 hours, then reports the hour with the
    shortest predicted wait (best) and longest predicted wait (worst).

    Args:
        target_date: The specific date to analyze.
    """
    day_name = target_date.strftime("%A")
    week_of_month = (target_date.day - 1) // 7 + 1

    # ── Holiday guard ──────────────────────────────────────────────────────────
    is_hol, _ = get_holiday_flags(target_date)
    if is_hol:
        holiday_name = get_holiday_name(target_date)
        print("\n" + "=" * 80)
        print(f"⛔ LTO CDO IS CLOSED ON {day_name.upper()} {target_date.strftime('%B %d, %Y')}")
        print(f"   Reason: {holiday_name}")
        print("   Please choose a different date (Mon–Sat, non-holiday).")
        print("=" * 80)
        return
    # ──────────────────────────────────────────────────────────────────────────

    print("\n" + "=" * 80)
    print(f"🔍 BEST TIME TO VISIT ON {day_name.upper()} {target_date.strftime('%B %d, %Y')}")
    print("=" * 80)

    # Collect predictions for all hours
    predictions = []
    for hour in range(8, 17):
        mc = predict_wait_time_monte_carlo(target_date, hour)
        predictions.append((hour, mc["mean"], mc["queue_mean"], mc["p10"], mc["p90"]))

    # Find the best (minimum wait) and worst (maximum wait) hours
    best = min(predictions, key=lambda x: x[1])   # Sort by wait time, pick lowest
    worst = max(predictions, key=lambda x: x[1])   # Sort by wait time, pick highest

    # Display the best time
    print("\n✅ BEST TIME TO VISIT:")
    print(f"   🕐 {best[0]:02d}:00")
    print(f"   ⏱️ Wait time: {best[1]:.0f} minutes")
    print(f"   📉 Likely range (P10-P90): {best[3]:.0f}-{best[4]:.0f} minutes")
    print(f"   👥 Expected queue: ~{best[2]:.0f} people")
    level, rec = get_congestion_level(best[1])
    print(f"   📊 {level}")
    print(f"   💡 {rec}")

    # Display the worst time
    print("\n⚠️ WORST TIME TO AVOID:")
    print(f"   🕐 {worst[0]:02d}:00")
    print(f"   ⏱️ Wait time: {worst[1]:.0f} minutes")
    print(f"   📈 Likely range (P10-P90): {worst[3]:.0f}-{worst[4]:.0f} minutes")
    print(f"   👥 Expected queue: ~{worst[2]:.0f} people")
    level, rec = get_congestion_level(worst[1])
    print(f"   📊 {level}")
    print(f"   💡 {rec}")


def parse_date_input(prompt):
    """Prompt the user for a date and validate it's a working day.

    Accepts:
    - "today" → uses current date
    - "YYYY-MM-DD" → parses the date string

    Rejects Sundays (LTO is closed on Sundays).
    Loops until valid input is provided.

    Args:
        prompt: The text to display when asking for input.

    Returns:
        A validated pandas Timestamp for a working day (Mon–Sat).
    """
    valid_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    while True:
        raw = input(prompt).strip()
        if raw.lower() == "today":
            d = pd.Timestamp.now().normalize()  # Current date, midnight
        else:
            try:
                d = pd.to_datetime(raw)
            except Exception:
                print("   ❌ Invalid format. Use YYYY-MM-DD or 'today'.")
                continue
        # Reject Sundays
        if d.strftime("%A") not in valid_days:
            print(f"   ❌ {d.strftime('%A')} is not a working day (Mon–Sat only).")
            continue
        return d


def main():
    """Main menu loop — the entry point for the prediction CLI.

    Displays a menu with 4 options and processes user input in a loop
    until the user chooses to exit.
    """
    print("\n" + "=" * 80)
    print("🏢 LTO CDO QUEUE PREDICTION SYSTEM")
    print("🌤️ Date-Aware Forecast Using Machine Learning + Historical Patterns")
    print("=" * 80)
    print("\n📊 This system uses:")
    print("   • 90 days of actual LTO CDO queue data")
    print("   • Week-of-month, seasonal, and holiday-aware patterns")
    print("   • ML model (R²=0.965) trained on real queue/wait time data")
    print(f"   • Monte Carlo uncertainty simulation ({MONTE_CARLO_RUNS} runs per hour)")
    print("   • Enter a specific date — different dates give different predictions\n")

    # Main interaction loop
    while True:
        print("\n" + "-" * 50)
        print("📋 OPTIONS:")
        print("   1. 📅 View weekly forecast (plan your week)")
        print("   2. ⏰ View specific date forecast (plan your day)")
        print("   3. 🔍 Find best time on a specific date")
        print("   4. ❌ Exit")
        print("-" * 50)

        choice = input("\n👉 Select option (1-4): ").strip()

        if choice == "1":
            # Weekly forecast: user provides any date in the target week
            target = parse_date_input("📅 Enter any date within the week you want (YYYY-MM-DD or 'today'): ")
            display_weekly_forecast(target)

        elif choice == "2":
            # Daily forecast: hour-by-hour breakdown for one date
            target = parse_date_input("📅 Enter date (YYYY-MM-DD or 'today'): ")
            display_daily_forecast(target)

        elif choice == "3":
            # Best time finder: which hour has the shortest wait?
            target = parse_date_input("📅 Enter date (YYYY-MM-DD or 'today'): ")
            find_best_time(target)

        elif choice == "4":
            # Exit the program
            print("\n" + "=" * 80)
            print("👋 Thank you for using LTO Queue Predictor!")
            print("💡 Predictions reflect week-of-month patterns from actual data.")
            print("🚗 Plan your visit during LOW congestion times for the best experience!")
            print("=" * 80)
            break

        else:
            print("\n❌ Invalid option. Please select 1-4")

        # Pause before showing the menu again
        input("\n⏎ Press Enter to continue...")
