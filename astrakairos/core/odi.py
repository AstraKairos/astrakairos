"""
    Script to calculate the Orbital Deviation Index (ODI).
    The ODI is designed to quantify how much (in arcseconds) a given measurement deviates
    from the predicted position based on current orbital elements.
"""

import numpy as np
import re
import numpy as np
from scipy.optimize import newton

# Note: Should probably add proper error handling. For now tho (so far, testing), we just skip any lines that cause issues and move on.
# Note 2: Ideally, we'd also do some uncertainty propagation, even if it's just using standard error propagation formulas. This will
# be attempted if we start using the whole WDS measurements catalog, but we are yet to be confirmed that we're allowed to by the USNO.
# Note 3: In the future, the parsing functions will be moved to the data section of AstraKairos.

def get_orb6_parameters(filepath, target_wds_id, target_component=None):
    """
    Retrieves the most recent orbit for a given WDS ID.
    
    - Parse all matching lines in the file.
    - Attempt to extract the Reference Year (e.g., 'Pop1996b' -> 1996).
    - Return the orbit with the most recent publication year.
    """
    
    candidates = []

    with open(filepath, 'r', encoding='latin-1') as f:
        for line in f:
            if len(line) < 180 or not line[0].isdigit():
                continue

            # Match ID (Fixed Width)
            if line[19:29].strip() != target_wds_id.strip():
                continue
            
            # Match Component (there can be multiple orbits for different components, e.g., AB, AC)
            if target_component and target_component not in line:
                continue

            # Parse Parameters
            parts = line[80:].split()
            if len(parts) < 18:
                continue

            idx = 0
            current_orbit = {}
            
            try:
                # PERIOD (P)
                p_str = parts[idx]
                P = float(p_str.rstrip('cdhmyamudy'))
                
                p_unit = 'y'
                idx += 1
                if parts[idx][0].isalpha():
                    p_unit = parts[idx]
                    idx += 1
                idx += 1
                
                # Unit Conversion (to years)
                if 'c' in p_str or 'c' in p_unit: P *= 100.0
                elif 'd' in p_str or 'd' in p_unit: P /= 365.25
                elif 'h' in p_str or 'h' in p_unit: P /= 8766.0
                elif 'm' in p_str or 'm' in p_unit: P /= 525960.0
                current_orbit['P'] = P

                # SEMI-MAJOR AXIS (a)
                a_str = parts[idx]
                a = float(a_str.rstrip('amu'))
                
                a_unit = 'a'
                idx += 1
                if parts[idx][0].isalpha():
                    a_unit = parts[idx]
                    idx += 1
                idx += 1
                
                if 'm' in a_str or 'm' in a_unit: a /= 1000.0
                elif 'u' in a_str or 'u' in a_unit: a /= 1e6
                current_orbit['a'] = a

                # INCLINATION (i)
                if parts[idx] == '.': raise ValueError("Missing i")
                current_orbit['i'] = float(parts[idx])
                idx += 2

                # NODE (Omega)
                if parts[idx] == '.': raise ValueError("Missing Omega")
                current_orbit['Omega'] = float(parts[idx].rstrip('*'))
                idx += 2

                # TIME OF PERIASTRON (T)
                t_str = parts[idx]
                if t_str == '.': raise ValueError("Missing T")
                T = float(t_str.rstrip('dmy'))
                
                t_unit = 'y'
                idx += 1
                if parts[idx][0].isalpha():
                    t_unit = parts[idx]
                    idx += 1
                idx += 1

                # T Conversion
                is_jd = 'd' in t_str or 'd' in t_unit
                is_mjd = 'm' in t_str or 'm' in t_unit
                
                if is_jd: T = 2000.0 + ((2400000.0 + T) - 2451545.0) / 365.25
                elif is_mjd: T = 2000.0 + ((2400000.5 + T) - 2451545.0) / 365.25
                current_orbit['T'] = T

                # ECCENTRICITY (e)
                if parts[idx] == '.': raise ValueError("Missing e")
                current_orbit['e'] = float(parts[idx])
                idx += 2

                # ARGUMENT OF PERIASTRON (w)
                if parts[idx] == '.': raise ValueError("Missing w")
                current_orbit['omega'] = float(parts[idx])
                
                # EXTRACT REFERENCE YEAR (to select the most recent orbit if multiple are available)
                # The reference is usually the second to last item, or matches AuthorYYYY
                # We search the whole line text for the standard Ref format to be safe
                # Regex looks for 4 digits 1800-2099 inside a word
                ref_year = 0
                ref_match = re.search(r'([1-2][0-89][0-9]{2})[a-z]?', line[150:]) # Search end of line
                if ref_match:
                    ref_year = int(ref_match.group(1))
                
                current_orbit['ref_year'] = ref_year
                
                # Orbit's Grade (Usually between 1-5, but apparently it can be up to 9)
                grade = 5
                for k in range(len(parts)-5, len(parts)):
                    if parts[k].isdigit() and 1 <= int(parts[k]) <= 9:
                        grade = int(parts[k])
                        break
                current_orbit['grade'] = grade
                
                # Add to candidates
                candidates.append(current_orbit)

            except (ValueError, IndexError):
                continue

    # Can't use any orbit if we didn't find any valid candidates
    if not candidates:
        return False
        
    # Sort candidates by Reference Year (Descending)
    # If years are equal, could sort by Grade (Ascending)
    candidates.sort(key=lambda x: (x['ref_year'], -x['grade']), reverse=True)
    
    return candidates[0]

def get_wds_measurement(file_path, target_wds_id, target_components=None):
    """
    Retrieves the latest position angle (theta) and separation (rho) 
    from a WDS summary line.
    """
    
    # Following the official USNO format...
    # 1-10: WDS Coordinates (ID) -> slice(0, 10)
    S_ID = slice(0, 10)
    # 11-17: Discoverer -> slice(10, 17)
    S_DISC = slice(10, 17)
    # 18-22: Components -> slice(17, 22)
    S_COMP = slice(17, 22)
    # 29-32: Date (last) -> slice(28, 32)
    S_DATE = slice(28, 32)
    # 43-45: Position Angle (last) -> slice(42, 45)
    S_THETA = slice(42, 45)
    # 53-57: Separation (last) -> slice(52, 57)
    S_RHO = slice(52, 57)

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if len(line) < 60:
                continue

            # Extract the WDS ID from the line
            line_id = line[S_ID].strip()

            if line_id == target_wds_id.strip():
                
                # Extract components
                line_comps = line[S_COMP].strip()
                
                # Filter by component if specified (e.g., AB, AC)
                if target_components:
                    if line_comps != target_components.strip():
                        continue
                
                # Parse the latest data
                try:
                    date_str = line[S_DATE].strip()
                    theta_str = line[S_THETA].strip()
                    rho_str = line[S_RHO].strip()

                    # Handle missing data (WDS uses '-1' for missing data)
                    date_last = int(date_str) if date_str and date_str != '-1' else None
                    theta_last = float(theta_str) if theta_str and theta_str != '-1' else None
                    rho_last = float(rho_str) if rho_str and rho_str != '-1' else None
                    
                    return {
                        'wds_id': line_id,
                        'discoverer': line[S_DISC].strip(),
                        'components': line_comps,
                        'date_last': date_last,
                        'theta': theta_last, # deg
                        'rho': rho_last # arcsec
                    }

                except ValueError:
                    continue

    return None

def binary_star_position(t, P, T, e, a, i, omega, Omega):
    """
    Calculates the separation (rho) and position angle (theta) of a binary star.
    t: Epoch of the position to predict; the rest are orbital parameters that should be
    directly taken from the ORB6 catalog (in theory, this should work for any orbit tho).
    """
    
    # Kepler's equation solver using Newton-Raphson's method.
    # There are technically better methods depending on the excentricity (very high, or e=0), but for now, this should be sufficient
    # for the sake of testing our proof of concept (and the majority of ORB6 systems).

    i_rad = np.radians(i)
    omega_rad = np.radians(omega)
    Omega_rad = np.radians(Omega)
    
    n = 2 * np.pi / P
    M = n * (t - T)
    
    M = (M + np.pi) % (2 * np.pi) - np.pi
    
    def kepler_eq(E, M_val):
        return E - e * np.sin(E) - M_val

    if np.isscalar(M):
        E = newton(kepler_eq, M, args=(M,))
    else:
        E = np.array([newton(kepler_eq, m, args=(m,)) for m in M])

    # Thiele-Innes' constants
    A = a * (np.cos(omega_rad) * np.cos(Omega_rad) - np.sin(omega_rad) * np.sin(Omega_rad) * np.cos(i_rad))
    B = a * (np.cos(omega_rad) * np.sin(Omega_rad) + np.sin(omega_rad) * np.cos(Omega_rad) * np.cos(i_rad))
    F = a * (-np.sin(omega_rad) * np.cos(Omega_rad) - np.cos(omega_rad) * np.sin(Omega_rad) * np.cos(i_rad))
    G = a * (-np.sin(omega_rad) * np.sin(Omega_rad) + np.cos(omega_rad) * np.cos(Omega_rad) * np.cos(i_rad))
    
    X_pos = A * (np.cos(E) - e) + F * np.sqrt(1 - e**2) * np.sin(E)
    Y_pos = B * (np.cos(E) - e) + G * np.sqrt(1 - e**2) * np.sin(E)
    
    rho = np.sqrt(X_pos**2 + Y_pos**2)
    
    # There's this issue with Speckle Interferometry where the position angle can be flipped by 180° due to quadrant ambiguity.
    # To solve this, we calculate the ODI using both angles, and pick the lowest one as the final ODI for that system.
    theta_rad = np.arctan2(Y_pos, X_pos)
    theta = np.degrees(theta_rad) % 360
    theta_flipped = theta # For angles between 0-180, and 180-360
    if 180 <= theta < 360:
        theta_flipped = (theta - 180) % 360
    else:
        theta_flipped = (theta + 180) % 360
    
    return rho, theta, theta_flipped
    
def calculate_deviation(observed_rho, observed_theta, predicted_rho, predicted_theta):
    """
    Basic calculation of the angular deviation between observed and predicted positions.
    """
    delta_rho = observed_rho - predicted_rho
    delta_theta = np.radians(observed_theta - predicted_theta)
    
    deviation = np.sqrt(delta_rho**2 + (predicted_rho * delta_theta)**2)
    
    return deviation

def run_odi_calculation(wds_id, component, orb6_file, wds_summary_file):
    """
    Calculates the ODI for a given system.
    RETURNS: (best_odi, components) or None
    best_odi stands for the lowest ODI between the normal and flipped position angle predictions (+- 180°).
    """

    orb_params = get_orb6_parameters(orb6_file, wds_id, component)
    meas_data = get_wds_measurement(wds_summary_file, wds_id, component)
    
    if not orb_params:
        print(f"No valid orbit found for {wds_id}")
        return None
    if not meas_data:
        print(f"No measurement data found for {wds_id}")
        return None

    # Orbital parameters
    P = orb_params['P']
    T = orb_params['T']
    e = orb_params['e']
    a = orb_params['a']
    i = orb_params['i']
    omega = orb_params['omega']
    Omega = orb_params['Omega']
    # Latest measurement data
    date_last = meas_data['date_last']
    theta_last = meas_data['theta']
    rho_last = meas_data['rho']
    
    # Validate data existence before calculation
    if date_last is None or theta_last is None or rho_last is None or date_last == '-1' or theta_last == '-1' or rho_last == '-1':
        return None

    # Predicted measurement:
    predicted_rho, predicted_theta, predicted_theta_flipped = binary_star_position(date_last, P, T, e, a, i, omega, Omega)
    
    # Calculate deviation
    deviation = calculate_deviation(rho_last, theta_last, predicted_rho, predicted_theta)
    deviation_flipped = calculate_deviation(rho_last, theta_last, predicted_rho, predicted_theta_flipped)
    
    # Select the minimum deviation (best fit considering quadrant ambiguity)
    best_odi = min(deviation, deviation_flipped)

    print(f"WDS ID: {wds_id} | Comp: {meas_data['components']} | Ref: {orb_params['ref_year']}")
    print(f"Observed ({date_last}): rho={rho_last:.3f}\", theta={theta_last:.2f}°")
    print(f"Predicted: rho={predicted_rho:.3f}\", theta={predicted_theta:.2f}°")
    print(f"ODI: {deviation:.4f} | ODI (Flipped): {deviation_flipped:.4f}")
    print(f"Best ODI (Selected): {best_odi:.4f}")
    print("-----------------------------------------")

    # Return tuple with best_odi AND the component considered
    return best_odi, meas_data['components']

def process_full_catalog(orb6_file, wds_summary_file, output_filename="odi_results.txt"):
    """
    For statistical analysis, we also add a function to process the entire ORB6 catalog.

    - Loads all ORB6 data into memory (Dictionary). Note that this works too for a smaller number of orbits,
    just need to manually cut/edit the text file.
    - Loads all WDS summary data into memory (Dictionary).
    - Iterates to calculate the ODI for all systems.
    """
    print(f"Loading orbital data from {orb6_file}...")
    
    # Get orbital parameters
    orbits_db = {} 

    with open(orb6_file, 'r', encoding='latin-1') as f:
        for line in f:
            if len(line) < 180 or not line[0].isdigit():
                continue

            wds_id = line[19:29].strip()
            
            parts = line[80:].split()
            if len(parts) < 18: continue

            idx = 0
            current_orbit = {}

            try:
                # P
                p_str = parts[idx]
                P = float(p_str.rstrip('cdhmyamudy'))
                p_unit = 'y'
                idx += 1
                if parts[idx][0].isalpha(): p_unit = parts[idx]; idx += 1
                idx += 1
                # Convert P to years
                if 'c' in p_str or 'c' in p_unit: P *= 100.0
                elif 'd' in p_str or 'd' in p_unit: P /= 365.25
                elif 'h' in p_str or 'h' in p_unit: P /= 8766.0
                elif 'm' in p_str or 'm' in p_unit: P /= 525960.0
                current_orbit['P'] = P

                # a
                a_str = parts[idx]
                a = float(a_str.rstrip('amu'))
                a_unit = 'a'
                idx += 1
                if parts[idx][0].isalpha(): a_unit = parts[idx]; idx += 1
                idx += 1
                if 'm' in a_str or 'm' in a_unit: a /= 1000.0
                elif 'u' in a_str or 'u' in a_unit: a /= 1e6
                current_orbit['a'] = a

                # i, Omega, T, e, omega
                if parts[idx] == '.': continue
                current_orbit['i'] = float(parts[idx]); idx += 2
                
                if parts[idx] == '.': continue
                current_orbit['Omega'] = float(parts[idx].rstrip('*')); idx += 2
                
                t_str = parts[idx]
                if t_str == '.': continue
                T = float(t_str.rstrip('dmy'))
                t_unit = 'y'
                idx += 1
                if parts[idx][0].isalpha(): t_unit = parts[idx]; idx += 1
                idx += 1
                is_jd = 'd' in t_str or 'd' in t_unit
                is_mjd = 'm' in t_str or 'm' in t_unit
                if is_jd: T = 2000.0 + ((2400000.0 + T) - 2451545.0) / 365.25
                elif is_mjd: T = 2000.0 + ((2400000.5 + T) - 2451545.0) / 365.25
                current_orbit['T'] = T

                if parts[idx] == '.': continue
                current_orbit['e'] = float(parts[idx]); idx += 2
                
                if parts[idx] == '.': continue
                current_orbit['omega'] = float(parts[idx])
                
                # Ref Year & Grade
                ref_year = 0
                ref_match = re.search(r'([1-2][0-89][0-9]{2})[a-z]?', line[150:]) 
                if ref_match: ref_year = int(ref_match.group(1))
                current_orbit['ref_year'] = ref_year
                
                grade = 5
                for k in range(len(parts)-5, len(parts)):
                    if parts[k].isdigit() and 1 <= int(parts[k]) <= 9:
                        grade = int(parts[k])
                        break
                current_orbit['grade'] = grade

                # If ID exists, keep the one with higher Year, then lower Grade (better quality)
                if wds_id in orbits_db:
                    prev = orbits_db[wds_id]
                    # Logic: New year > Old year OR (Year equal AND New Grade < Old Grade)
                    is_better = False
                    if current_orbit['ref_year'] > prev['ref_year']:
                        is_better = True
                    elif current_orbit['ref_year'] == prev['ref_year']:
                        if current_orbit['grade'] < prev['grade']:
                            is_better = True
                    
                    if is_better:
                        orbits_db[wds_id] = current_orbit
                else:
                    orbits_db[wds_id] = current_orbit

            except (ValueError, IndexError):
                continue

    print(f"Loaded {len(orbits_db)} unique orbital systems into memory.")
    print(f"Loading measurement data from {wds_summary_file}...")

    # Get astrometric measurements
    measurements_db = {}
    
    S_ID = slice(0, 10); S_COMP = slice(17, 22); S_DATE = slice(28, 32)
    S_THETA = slice(42, 45); S_RHO = slice(52, 57)

    with open(wds_summary_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if len(line) < 60: continue
            
            line_id = line[S_ID].strip()
            # If we don't have an orbit for this ID
            if line_id not in orbits_db:
                continue

            try:
                date_str = line[S_DATE].strip()
                theta_str = line[S_THETA].strip()
                rho_str = line[S_RHO].strip()
                
                if not date_str or date_str == '.' or not theta_str or theta_str == '.' or not rho_str or rho_str == '.':
                    continue

                meas = {
                    'components': line[S_COMP].strip(),
                    'date_last': int(date_str),
                    'theta': float(theta_str),
                    'rho': float(rho_str)
                }
                
                if line_id not in measurements_db:
                    measurements_db[line_id] = meas
            except ValueError:
                continue

    print(f"Loaded measurements for {len(measurements_db)} matching systems.")
    print(f"Calculating ODI and saving to {output_filename}...")

    # Calculations
    count = 0
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        outfile.write("WDS_ID\tComponent\tODI_Best\n")
        
        for wds_id, orbit in orbits_db.items():
            if wds_id in measurements_db:
                meas = measurements_db[wds_id]
                
                try:
                    pred_rho, pred_theta, pred_theta_flip = binary_star_position(
                        meas['date_last'], orbit['P'], orbit['T'], orbit['e'], 
                        orbit['a'], orbit['i'], orbit['omega'], orbit['Omega']
                    )
                    
                    dev = calculate_deviation(meas['rho'], meas['theta'], pred_rho, pred_theta)
                    dev_flip = calculate_deviation(meas['rho'], meas['theta'], pred_rho, pred_theta_flip)
                    
                    best_odi = min(dev, dev_flip)
                    
                    outfile.write(f"{wds_id}\t{meas['components']}\t{best_odi:.4f}\n")
                    count += 1
                except Exception:
                    continue

    print(f"\nBatch processing complete. Successfully processed {count} systems.")

# Example usage:
if __name__ == "__main__":
    orb6_file = "data/orb6.txt"
    wds_summary_file = "data/wdsweb_summ2.txt"
    output_file = "odi_results.txt"

    # Run for a single system
    # wds_id = "00026+1841" 
    # run_odi_calculation(wds_id, None, orb6_file, wds_summary_file)

    # Run for the full catalog
    process_full_catalog(orb6_file, wds_summary_file, output_file)