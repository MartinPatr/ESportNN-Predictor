import pandas as pd

# Assuming df is your DataFrame
df = pd.read_csv('LCKStatsRaw.csv', dtype='str')

def get_player_champions():
    # Iterate through the rows
    red_champs = []
    blue_champs = []
    red_index = 0
    blue_index = 0
    previous_match = ""
    for index, row in df.iterrows():
        # Check if the match ID is the same as the previous row
        if row['gameid'] != previous_match and index != 0:
            # Add the champion names to the new column
            for i, champ in enumerate(red_champs):
                df.at[red_index,f'Champion_{i+1}'] = champ
            for i, champ in enumerate(blue_champs):
                df.at[blue_index,f'Champion_{i+1}'] = champ        
            red_champs = []
            blue_champs = []
        # Check if participant IDs are 200 or 100
        if row['participantid'] == "200":
            red_index = index
        elif row['participantid'] == "100":
            blue_index = index
        else:
            champion = row['champion']
            if row['side'] == 'Red':
                red_champs.append(champion)
            elif row["side"] == 'Blue':
                blue_champs.append(champion)
            else:
                print("Error")
        previous_match = row['gameid']

get_player_champions()


# Delete non-team columns
df = df[df["position"] == "team"]

# Now df contains the additional opponent information
df.to_csv('LCKStatsFiltered.csv', index=False)

df = pd.read_csv('LCKStatsFiltered.csv', dtype='str')

# To get the opponent's data, we need to create a new DataFrame
last_team = False
for i in range(0, len(df)):
    if last_team:
        index_add = -1
        last_team = False
    else:
        index_add = 1
        last_team = True
    # Add opponent's team name to the current row
    df.loc[i, 'opponent_teamname'] = df.loc[i + index_add, "teamname"]


df.to_csv('LCKStatsFiltered.csv', index=False)

# Create dictionaries to map champions and teams to numbers
champion_mapping = {}
team_mapping_dic = {}
# Define a function to map 
def map_team(team, team_mapping=team_mapping_dic):
    if team not in team_mapping:
        team_mapping[team] = len(team_mapping) + 1
    return team_mapping[team]

# Map champions to numbers and create new columns
for i in range(1, 6):
    df[f'Champion_{i}_Number'] = df[f'Champion_{i}'].apply(map_team, args=(champion_mapping,))

# Map banned champions to numbers and create new columns
for i in range(1, 6):
    df[f'Champion_Banned_Number{i}'] = df[f'ban{i}'].apply(map_team, args=(champion_mapping,))  

# Apply the team mapping to both 'opponent_teamid' and 'opponent_teamname' columns
df['Team_Number'] = df['teamname'].apply(map_team)
df['Opponent_Team_Number'] = df['opponent_teamname'].apply(map_team)

# Apply the team mapping to both 'opponent_teamid' and 'opponent_teamname' columns
df['Side_Number'] = df['side'].apply(lambda x: 0 if x == 'Red' else 1)


# To get the opponent's data, we need to create a new DataFrame
last_team = False
for index, row in df.iterrows():
    if last_team:
        index_add = -1
        last_team = False
    else:
        index_add = 1
        last_team = True

    # Map opponent's champions to numbers
    for i in range(1, 6):
        df.loc[index,f'opponent_Champion_{i}_Number'] = df.loc[index + index_add, f'Champion_{i}_Number']

    # Map opponent's bans to numbers
    for i in range(1, 6):
        df.loc[index,f'opponent_Champion_Banned_Number{i}'] = df.loc[index + index_add, f'Champion_Banned_Number{i}']


# Function to attempt conversion to integer
def try_convert(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return value

# Apply the conversion function to each element in the DataFrame
df = df.applymap(try_convert)


# Now df contains the additional opponent information
df.to_csv('LCKStatsFiltered.csv', index=False)
