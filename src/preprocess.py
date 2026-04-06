import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    
    # Ensure date is properly parsed
    df['date'] = pd.to_datetime(df['date'])
    
    # Derive week_of_month (1-4) so Week-1 Monday differs from Week-4 Monday
    df['week_of_month'] = df['date'].dt.day.apply(lambda d: (d - 1) // 7 + 1)
    
    # Remove any invalid rows
    df = df[df['waiting_time_min'] >= 0]
    df = df[df['queue_length_at_arrival'] >= 0]
    
    df = df.dropna()
    
    print(f"✅ Loaded {len(df)} records")
    print(f"📊 Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    print("\n📅 Data distribution by day:")
    day_counts = df['day_name'].value_counts()
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']:
        if day in day_counts.index:
            print(f"   {day}: {day_counts[day]} records")
    
    return df

def get_features(df):
    # Features in the correct order for the model
    features = [
        'hour',
        'day_of_week',
        'week_of_month',
        'is_peak_day',
        'queue_length_at_arrival',
        'service_time_min',
        'is_weekend',
        'is_peak_hour',
        'queue_length_lag1',
        'waiting_time_lag1'
    ]
    
    X = df[features]
    y = df['waiting_time_min']
    
    print(f"\n📊 Feature statistics:")
    print(f"   Peak days (Mon/Fri): {X['is_peak_day'].sum()} records")
    print(f"   Non-peak days: {len(X) - X['is_peak_day'].sum()} records")
    print(f"   Peak hours: {X['is_peak_hour'].sum()} records")
    print(f"   Week-of-month range: {X['week_of_month'].min()} to {X['week_of_month'].max()}")
    
    return X, y, features