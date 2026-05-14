import pandas as pd

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


def build_pattern_maps(df, value_col):
    day_hour = {}
    day_week_hour = {}
    day_month_hour = {}
    day_month_week_hour = {}

    for day in DAYS:
        day_df = df[df["day_name"] == day]
        day_hour[day] = {}
        day_week_hour[day] = {}
        day_month_hour[day] = {}
        day_month_week_hour[day] = {}

        for hour in range(8, 17):
            day_hour[day][hour] = day_df[day_df["hour"] == hour][value_col].mean()

        for week in range(1, 6):
            week_df = day_df[day_df["week_of_month"] == week]
            day_week_hour[day][week] = {}
            for hour in range(8, 17):
                day_week_hour[day][week][hour] = week_df[week_df["hour"] == hour][value_col].mean()

        for month in range(1, 13):
            month_df = day_df[day_df["month"] == month]
            day_month_hour[day][month] = {}
            for hour in range(8, 17):
                day_month_hour[day][month][hour] = month_df[month_df["hour"] == hour][value_col].mean()

            day_month_week_hour[day][month] = {}
            for week in range(1, 6):
                week_df = month_df[month_df["week_of_month"] == week]
                day_month_week_hour[day][month][week] = {}
                for hour in range(8, 17):
                    day_month_week_hour[day][month][week][hour] = week_df[week_df["hour"] == hour][value_col].mean()

    return {
        "day_hour": day_hour,
        "day_week_hour": day_week_hour,
        "day_month_hour": day_month_hour,
        "day_month_week_hour": day_month_week_hour,
    }


def get_pattern_value(pattern_maps, day_name, month, week, hour, default_value):
    value = pattern_maps["day_month_week_hour"][day_name][month][week][hour]
    if not pd.isna(value):
        return float(value)

    value = pattern_maps["day_month_hour"][day_name][month][hour]
    if not pd.isna(value):
        return float(value)

    value = pattern_maps["day_week_hour"][day_name][week][hour]
    if not pd.isna(value):
        return float(value)

    value = pattern_maps["day_hour"][day_name][hour]
    if not pd.isna(value):
        return float(value)

    return float(default_value)
