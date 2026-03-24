#!/usr/bin/env python3
"""
Explore AddBiomechanics dataset: scan .b3d files, extract metadata,
and filter for clean walking trials with GRF data.

Usage:
    conda activate opensim
    python explore_addbiomechanics.py /path/to/addbiomechanics/

Output: CSV summary + filtered list of subjects suitable for multi-subject MocoTrack.
"""

import nimblephysics as nimble
import os
import sys
import csv
import numpy as np
from pathlib import Path

def explore_b3d(b3d_path):
    """Extract metadata from a single .b3d file."""
    try:
        subject = nimble.biomechanics.SubjectOnDisk(str(b3d_path))
    except Exception as e:
        return {'file': str(b3d_path), 'error': str(e)}

    info = {
        'file': str(b3d_path),
        'filename': os.path.basename(b3d_path),
        'mass_kg': None,
        'height_m': None,
        'age_years': None,
        'bio_sex': None,
        'num_dofs': None,
        'num_trials': None,
        'num_force_plates': None,
        'quality': None,
        'trial_names': [],
        'trial_lengths': [],
        'trial_timesteps': [],
        'has_grf': False,
        'total_frames': 0,
        'walking_trials': [],
        'tags': [],
        'notes': '',
        'error': None,
    }

    try:
        info['mass_kg'] = subject.getMassKg()
        info['height_m'] = subject.getHeightM()
        info['num_dofs'] = subject.getNumDofs()
        info['num_trials'] = subject.getNumTrials()
        info['num_force_plates'] = subject.getNumForcePlates()
    except:
        pass

    try:
        info['age_years'] = subject.getAgeYears()
    except:
        pass

    try:
        info['bio_sex'] = subject.getBiologicalSex()
    except:
        pass

    try:
        info['tags'] = list(subject.getSubjectTags())
    except:
        pass

    try:
        info['notes'] = subject.getNotes()
    except:
        pass

    # Check quality (lower residuals = better)
    try:
        # Get linear and angular residual norms per trial
        for t in range(subject.getNumTrials()):
            trial_name = subject.getTrialName(t)
            trial_len = subject.getTrialLength(t)
            trial_dt = subject.getTrialTimestep(t)

            info['trial_names'].append(trial_name)
            info['trial_lengths'].append(trial_len)
            info['trial_timesteps'].append(trial_dt)
            info['total_frames'] += trial_len

            # Check if trial has GRF by looking at missing GRF flags
            has_grf_this_trial = False
            try:
                missing_grf = subject.getMissingGRF(t)
                # missing_grf is a list of MissingGRFReason per frame
                # If most frames are NOT missing GRF, the trial has GRF
                not_missing = sum(1 for m in missing_grf 
                                  if m == nimble.biomechanics.MissingGRFReason.notMissingGRF)
                if not_missing > trial_len * 0.5:
                    has_grf_this_trial = True
                    info['has_grf'] = True
            except:
                pass

            # Check trial name for walking keywords
            name_lower = trial_name.lower() if trial_name else ''
            walk_keywords = ['walk', 'gait', 'treadmill', 'overground', 'level']
            is_walking = any(kw in name_lower for kw in walk_keywords)

            # Also check trial tags
            try:
                trial_tags = subject.getTrialTags(t)
                tag_str = ' '.join(trial_tags).lower()
                is_walking = is_walking or any(kw in tag_str for kw in walk_keywords)
            except:
                pass

            if has_grf_this_trial:
                info['walking_trials'].append({
                    'index': t,
                    'name': trial_name,
                    'length': trial_len,
                    'timestep': trial_dt,
                    'duration_s': trial_len * trial_dt if trial_dt else None,
                    'name_suggests_walking': is_walking,
                })

    except Exception as e:
        info['error'] = str(e)

    return info


def main():
    if len(sys.argv) < 2:
        data_dir = os.path.expanduser(
            '~/repos/projects/exo-assist-pipeline/data/addbiomechanics/')
    else:
        data_dir = sys.argv[1]

    # Find all .b3d files
    b3d_files = sorted(Path(data_dir).rglob('*.b3d'))
    print(f"Found {len(b3d_files)} .b3d files in {data_dir}")

    if len(b3d_files) == 0:
        print("No .b3d files found!")
        sys.exit(1)

    # Scan all files
    results = []
    errors = []

    for i, b3d_path in enumerate(b3d_files):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Scanning {i+1}/{len(b3d_files)}: {b3d_path.name}")
        
        info = explore_b3d(str(b3d_path))
        
        if info.get('error'):
            errors.append(info)
        else:
            results.append(info)

    # Summary
    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total .b3d files scanned: {len(b3d_files)}")
    print(f"Successfully loaded: {len(results)}")
    print(f"Errors: {len(errors)}")

    # Subjects with GRF
    with_grf = [r for r in results if r['has_grf']]
    print(f"\nSubjects with GRF data: {len(with_grf)}")

    # Subjects with walking trials + GRF
    with_walking_grf = [r for r in results 
                        if any(t['name_suggests_walking'] for t in r['walking_trials'])]
    print(f"Subjects with walking+GRF trials: {len(with_walking_grf)}")

    # Body stats
    masses = [r['mass_kg'] for r in results if r['mass_kg'] and r['mass_kg'] > 0]
    heights = [r['height_m'] for r in results if r['height_m'] and r['height_m'] > 0]
    
    if masses:
        print(f"\nBody mass: {np.mean(masses):.1f} ± {np.std(masses):.1f} kg "
              f"(range: {np.min(masses):.1f}–{np.max(masses):.1f})")
    if heights:
        print(f"Height:    {np.mean(heights):.2f} ± {np.std(heights):.2f} m "
              f"(range: {np.min(heights):.2f}–{np.max(heights):.2f})")

    # Total frames
    total_frames = sum(r['total_frames'] for r in results)
    print(f"\nTotal frames: {total_frames:,}")
    print(f"Total trials: {sum(r['num_trials'] or 0 for r in results)}")

    # DOF distribution
    dofs = set(r['num_dofs'] for r in results if r['num_dofs'])
    print(f"Unique DOF counts: {sorted(dofs)}")

    # Write CSV summary
    output_dir = os.path.expanduser(
        '~/repos/projects/exo-assist-pipeline/data/addbiomechanics/')
    csv_path = os.path.join(output_dir, 'dataset_summary.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'filename', 'mass_kg', 'height_m', 'age_years', 'bio_sex',
            'num_dofs', 'num_trials', 'num_force_plates', 'total_frames',
            'has_grf', 'num_grf_trials', 'has_walking_grf', 'tags', 'notes'
        ])
        for r in results:
            has_walking = any(t['name_suggests_walking'] for t in r['walking_trials'])
            writer.writerow([
                r['filename'],
                f"{r['mass_kg']:.1f}" if r['mass_kg'] else '',
                f"{r['height_m']:.2f}" if r['height_m'] else '',
                r['age_years'] if r['age_years'] else '',
                r['bio_sex'] if r['bio_sex'] else '',
                r['num_dofs'],
                r['num_trials'],
                r['num_force_plates'],
                r['total_frames'],
                r['has_grf'],
                len(r['walking_trials']),
                has_walking,
                ';'.join(r['tags']) if r['tags'] else '',
                r['notes'][:100] if r['notes'] else '',
            ])

    print(f"\nCSV saved to: {csv_path}")

    # Write filtered list of good walking candidates
    candidates_path = os.path.join(output_dir, 'walking_candidates.csv')
    with open(candidates_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'filename', 'mass_kg', 'height_m', 'num_dofs',
            'trial_index', 'trial_name', 'trial_frames', 'trial_duration_s'
        ])
        for r in with_grf:
            for trial in r['walking_trials']:
                writer.writerow([
                    r['filename'],
                    f"{r['mass_kg']:.1f}" if r['mass_kg'] else '',
                    f"{r['height_m']:.2f}" if r['height_m'] else '',
                    r['num_dofs'],
                    trial['index'],
                    trial['name'],
                    trial['length'],
                    f"{trial['duration_s']:.2f}" if trial['duration_s'] else '',
                ])

    print(f"Walking candidates saved to: {candidates_path}")

    # Print top 10 candidates (sorted by trial duration)
    print(f"\n{'='*60}")
    print(f"TOP WALKING CANDIDATES (longest trials with GRF)")
    print(f"{'='*60}")
    
    all_candidates = []
    for r in with_grf:
        for trial in r['walking_trials']:
            all_candidates.append({
                'file': r['filename'],
                'mass': r['mass_kg'],
                'height': r['height_m'],
                'dofs': r['num_dofs'],
                **trial
            })
    
    all_candidates.sort(key=lambda x: x.get('duration_s') or 0, reverse=True)
    
    for c in all_candidates[:20]:
        print(f"  {c['file']:50s} | {c['mass']:6.1f} kg | {c['height']:5.2f} m | "
              f"{c['dofs']} DOF | trial={c['name']:30s} | "
              f"{c.get('duration_s', 0):6.1f}s | {c['length']} frames")

    if errors:
        print(f"\n{'='*60}")
        print(f"ERRORS ({len(errors)} files)")
        print(f"{'='*60}")
        for e in errors[:10]:
            print(f"  {e['file']}: {e['error']}")


if __name__ == '__main__':
    main()
