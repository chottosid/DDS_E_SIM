import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from itertools import product
import edlib
import re
import os
from datetime import datetime

def calculate_kmer_rates(orig_seqs, output_seqs, k=3):
    bases = ['A', 'C', 'G', 'T']
    all_kmers = [''.join(p) for p in product(bases, repeat=k)]
    
    kmer_counts_orig = defaultdict(int)
    kmer_counts_output = defaultdict(int)
    
    for orig, out in zip(orig_seqs, output_seqs):
        for i in range(len(orig) - k + 1):
            kmer = orig[i:i+k]
            kmer_counts_orig[kmer] += 1
            
        for i in range(len(out) - k + 1):
            kmer = out[i:i+k]
            kmer_counts_output[kmer] += 1
    
    kmer_rates = {}
    for kmer in all_kmers:
        orig_count = kmer_counts_orig[kmer]
        out_count = kmer_counts_output[kmer]
        if orig_count > 0 or out_count > 0:
            kmer_rates[kmer] = {
                'original_count': orig_count,
                'output_count': out_count
            }
    
    return kmer_rates

def analyze_errors(orig_seqs, output_seqs):
    total_bases = sum(len(seq) for seq in orig_seqs)
    total_subs = 0
    total_ins = 0
    total_dels = 0
    sub_matrix = defaultdict(lambda: defaultdict(int))
    pos_errors = defaultdict(lambda: {'subs': 0, 'ins': 0, 'dels': 0, 'total': 0})
    
    max_len = 0
    
    for orig, out in zip(orig_seqs, output_seqs):
        result = edlib.align(orig, out, task="path", mode="NW")
        nice = edlib.getNiceAlignment(result, orig, out)
        
        aligned_orig = nice["query_aligned"]
        aligned_out = nice["target_aligned"]
        
        max_len = max(max_len, len(orig))
        
        pos = 0
        for i in range(len(aligned_orig)):
            orig_base = aligned_orig[i]
            out_base = aligned_out[i]
            
            if orig_base == '-':
                total_ins += 1
                pos_errors[pos]['ins'] += 1
            elif out_base == '-':
                total_dels += 1
                pos_errors[pos]['dels'] += 1
                pos += 1
            elif orig_base != out_base:
                total_subs += 1
                pos_errors[pos]['subs'] += 1
                sub_matrix[orig_base][out_base] += 1
                pos += 1
            else:
                pos += 1
            
            if orig_base != '-':
                pos_errors[pos-1]['total'] += 1

    total_errors = total_subs + total_ins + total_dels
    total_error_rate = total_errors / total_bases if total_bases > 0 else 0
    
    error_percentages = {
        'substitutions': (total_subs/total_errors*100 if total_errors > 0 else 0),
        'insertions': (total_ins/total_errors*100 if total_errors > 0 else 0),
        'deletions': (total_dels/total_errors*100 if total_errors > 0 else 0)
    }

    pos_rates = {pos: {
        'subs_rate': errors['subs']/errors['total'] if errors['total'] > 0 else 0,
        'ins_rate': errors['ins']/errors['total'] if errors['total'] > 0 else 0,
        'del_rate': errors['dels']/errors['total'] if errors['total'] > 0 else 0
    } for pos, errors in pos_errors.items()}

    bases = ['A', 'C', 'G', 'T']
    sub_rates = pd.DataFrame(0.0, index=bases, columns=bases)
    for orig in bases:
        total = sum(sub_matrix[orig].values())
        if total > 0:
            for new in bases:
                sub_rates.loc[orig, new] = float(sub_matrix[orig][new]) / total

    kmer_rates = calculate_kmer_rates(orig_seqs, output_seqs)

    return {
        'total_errors': total_errors,
        'total_bases': total_bases,
        'total_error_rate': total_error_rate,
        'error_counts': {'substitutions': total_subs, 'insertions': total_ins, 'deletions': total_dels},
        'error_percentages': error_percentages,
        'position_rates': pos_rates,
        'substitution_rates': sub_rates,
        'kmer_rates': kmer_rates,
        'max_seq_length': max_len
    }

def plot_error_analysis(results):
    fig = plt.figure(figsize=(20, 15))
    
    plt.subplot(3, 2, 1)
    plt.pie(results['error_percentages'].values(), 
            labels=[x.capitalize() for x in results['error_percentages'].keys()],
            autopct='%1.1f%%')
    plt.title('Distribution of Error Types')

    plt.subplot(3, 2, 2)
    pos = list(results['position_rates'].keys())
    subs_rates = [rates['subs_rate'] for rates in results['position_rates'].values()]
    ins_rates = [rates['ins_rate'] for rates in results['position_rates'].values()]
    del_rates = [rates['del_rate'] for rates in results['position_rates'].values()]

    plt.plot(pos, subs_rates, label='Substitutions')
    plt.plot(pos, ins_rates, label='Insertions')
    plt.plot(pos, del_rates, label='Deletions')
    plt.xlabel('Position')
    plt.ylabel('Error Rate')
    plt.title('Error Rates by Position')
    plt.legend()

    plt.subplot(3, 2, 3)
    sns.heatmap(results['substitution_rates'], 
                annot=True, 
                fmt='.2f', 
                cmap='Blues')
    plt.title('Substitution Rate Matrix')
    
    plt.subplot(3, 2, 4)
    kmer_df = pd.DataFrame([
        {'kmer': k, 'original_count': v['original_count'], 'output_count': v['output_count']}
        for k, v in results['kmer_rates'].items()
    ]).sort_values('original_count', ascending=False)
    
    x = np.arange(len(kmer_df))
    width = 0.35
    
    plt.bar(x - width/2, kmer_df['original_count'], width, label='Original')
    plt.bar(x + width/2, kmer_df['output_count'], width, label='Output')
    plt.xticks(x, kmer_df['kmer'], rotation=90)
    plt.xlabel('K-mer')
    plt.ylabel('Count')
    plt.title('K-mer Counts Comparison')
    plt.legend()
    
    plt.subplot(3, 2, 5)
    plt.bar(['Total Error Rate'], [results['total_error_rate']])
    plt.ylabel('Rate')
    plt.title('Overall Error Rate')
    
    plt.tight_layout()
    plt.show()

def parse_cigar(cigar):
    return [(int(count), op) for count, op in re.findall(r'(\d+)([=XID])', cigar)]

def calculate_error_rates(origs, comparison_seqs, max_length):
    n_seqs = len(origs)
    
    insertions = np.zeros(max_length)
    deletions = np.zeros(max_length)
    substitutions = np.zeros(max_length)
    valid_positions = np.zeros(max_length)
    
    for orig, comp_seq in zip(origs, comparison_seqs):
        for i in range(len(orig)):
            valid_positions[i] += 1
            
        alignment = edlib.align(orig, comp_seq, task="path", mode="global")
        
        orig_pos = 0
        comp_pos = 0
        for count, op in parse_cigar(alignment["cigar"]):
            if op == "=":
                orig_pos += count
                comp_pos += count
            elif op == "X":
                for i in range(count):
                    if orig_pos < len(orig):
                        substitutions[orig_pos] += 1
                    orig_pos += 1
                    comp_pos += 1
            elif op == "D":
                for i in range(count):
                    if comp_pos < len(comp_seq) and orig_pos < len(orig):
                        insertions[orig_pos] += 1
                    comp_pos += 1
            elif op == "I":
                for i in range(count):
                    if orig_pos < len(orig):
                        deletions[orig_pos] += 1
                    orig_pos += 1
    
    with np.errstate(divide='ignore', invalid='ignore'):
        insertions = np.where(valid_positions > 0, 
                            (insertions / valid_positions) * 100, 0)
        deletions = np.where(valid_positions > 0,
                           (deletions / valid_positions) * 100, 0)
        substitutions = np.where(valid_positions > 0,
                               (substitutions / valid_positions) * 100, 0)
    
    return {
        "insertion": insertions,
        "deletion": deletions,
        "substitution": substitutions
    }

def plot_error_rates(origs, reads, outputs, save_path="error_rates.pdf"):
    max_length = max(
        max(len(seq) for seq in origs),
        max(len(seq) for seq in reads),
        max(len(seq) for seq in outputs)
    )
    
    read_error_rates = calculate_error_rates(origs, reads, max_length)
    output_error_rates = calculate_error_rates(origs, outputs, max_length)
    
    x = np.arange(max_length)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    ax1.plot(x, read_error_rates["insertion"], '-', 
             color='blue', label='original sequences', linewidth=1)
    ax1.plot(x, output_error_rates["insertion"], '-',
             color='red', label='generated sequences', linewidth=1)
    ax1.set_xlabel('index')
    ax1.set_ylabel('proportion(%)')
    ax1.set_title('(a) insertion')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 2.0)
    
    ax2.plot(x, read_error_rates["substitution"], '-',
             color='blue', label='original sequences', linewidth=1)
    ax2.plot(x, output_error_rates["substitution"], '-',
             color='red', label='generated sequences', linewidth=1)
    ax2.set_xlabel('index')
    ax2.set_ylabel('proportion(%)')
    ax2.set_title('(b) substitution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 1.4)
    
    ax3.plot(x, read_error_rates["deletion"], '-',
             color='blue', label='original sequences', linewidth=1)
    ax3.plot(x, output_error_rates["deletion"], '-',
             color='red', label='generated sequences', linewidth=1)
    ax3.set_xlabel('index')
    ax3.set_ylabel('proportion(%)')
    ax3.set_title('(c) deletion')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(0.8, 1.6)
    
    plt.tight_layout()
    plt.show()
