import csv

# Set these whenever you want to run it, to where you're getting the RASaero txt files from and where you want the csv to go, right now its set up for my own local computer
Rocket = "SOLINV"
AEROfolders= "C:/Users/yosep/flight_sim/data/ras_alpha_files"
output_file = f"{AEROfolders}/converted_aerodynamics-{Rocket}.csv"

# These are the refrence values, a refrence point and Refrence length (THIS IS BODY DIAMETER), just use the same values as JARVIS
# This is for the rocket used to calculate its aerodyanmic values, They are hardcoded in for now
REFPOINT = 50.0 
L_ref = 6.0  

# These are how many alphas that i want to pull from RASAero
ALPHAS = list(range(0, 180))
# These are the index values for where each item is on the txt fiels from RASAero
IDX_MACH = 0
IDX_CD = 2
# After each header line RASAero writes, in order:
#   +1: Mach, CN alpha (0-4 deg), CP (0-4 deg)
#   +2: Mach, Alpha, CN potential, CN viscous
#   +3: Mach, Alpha, CN total, CP total
# CN and CP both come from the "total" line (CP is inches from the nose tip)
TOTAL_LINE_OFFSET = 3
IDX_CN = 2
IDX_CP = 3
MAX_MACH = 2.0   # we only need up to this mach. Without this there are like 15,000 lines in CSV, it slows things down       
STEP = 0.1 # This is how much the mach changes by each step, its meant to reduce how many points are in the CSV      

def fmt(x): #This is a tool gemini told me to use, it formats everything much more efficently thatn just changing it to a float, like i did before, and prevents things like "-0.0"
    """This builds a Compact number format, for example: 0, -0.449931, -2.34154 (removes -0 and 0.0)."""
    if x == 0:
        return "0"
    return f"{x:g}" # turns it into 6 digits

all_data = []

for alpha in ALPHAS:
    filename = f"{AEROfolders}/alpha{alpha}.txt" # this reads the appropiate file
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            # This just opens the file to be read
    except FileNotFoundError:
        print(f"Skipping alpha{alpha}.txt (File not found)")
        #This makes it so it skips a file if its not found, rather than trying to forcefully run it and crashing everything
        continue

    for i in range(len(lines) - TOTAL_LINE_OFFSET):
        line = lines[i].split()
        if len(line) < 12:
            continue
            # This finds the lines which are long enough to be header lines
        try:
            mach = float(line[IDX_MACH])
            CD = float(line[IDX_CD])
            totals = lines[i + TOTAL_LINE_OFFSET].split()
            CN = float(totals[IDX_CN])
            CP = float(totals[IDX_CP])
            # Using the header lines as refrence, we can just pull the rest of the infromation assuming that the placemnt i orignally set was correct
        except (ValueError, IndexError):
            continue

        if mach > MAX_MACH:
            continue
        if abs(mach / STEP - round(mach / STEP)) > 1e-6: #This prevents round off error! (This is exactly what i had to do in a 102 lab!)
            continue

        all_data.append((mach, alpha, CD, CN, CP))

# This reformats it in the way that our previous csv files have been formatted.
all_data.sort(key=lambda r: (r[0], r[1]))

# This recalculates everything into the correct format.
with open(output_file, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Mach", "Alpha", "Phi", "Cx", "Cy", "Cz", "CMx", "CMy", "CMz"])

    for mach, alpha, CD, CN, CP in all_data:
        Cx = -CD
        Cz = -CN
        CMy = 0.0 if CN == 0 else -CN * (CP - REFPOINT) / L_ref

        for phi in (0, 360):
            writer.writerow([fmt(mach), alpha, phi, fmt(Cx), 0, fmt(Cz), 0, fmt(CMy), 0])

print(f"Done with all {len(all_data)} points, saved to {output_file}")