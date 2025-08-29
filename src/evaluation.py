"""
DNA sequence error analysis tools
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import edlib
import re
from collections import defaultdict
from itertools import product
from typing import List, Dict, Tuple
import torch
from .utils import decode_one_hot_sequence, index_to_char


def calculate_kmer_rates(orig_seqs: List[str], output_seqs: List[str], k: int = 3) -> Dict:
    """Calculate k-mer frequencies and error rates."""
    # Generate all possible k-mers
    bases = ['A', 'C', 'G', 'T']
    all_kmers = [''.join(p) for p in product(bases, repeat=k)]
    
    # Initialize counters
    kmer_counts_orig = defaultdict(int)
    kmer_counts_output = defaultdict(int)
    
    # Count k-mers in original and output sequences
    for orig, out in zip(orig_seqs, output_seqs):
        # Count k-mers in original sequence
        for i in range(len(orig) - k + 1):
            kmer = orig[i:i+k]
            kmer_counts_orig[kmer] += 1
            
        # Count k-mers in output sequence
        for i in range(len(out) - k + 1):
            kmer = out[i:i+k]
            kmer_counts_output[kmer] += 1
    
    # Store counts for each k-mer
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


def analyze_errors(orig_seqs: List[str], output_seqs: List[str]) -> Dict:
    """Comprehensive error analysis using edlib alignment"""
    total_bases = sum(len(seq) for seq in orig_seqs)
    total_subs = 0
    total_ins = 0
    total_dels = 0
    sub_matrix = defaultdict(lambda: defaultdict(int))
    pos_errors = defaultdict(lambda: {'subs': 0, 'ins': 0, 'dels': 0, 'total': 0})
    
    # Keep track of max sequence length for position analysis
    max_len = 0
    
    for orig, out in zip(orig_seqs, output_seqs):
        # Get alignment using edlib
        result = edlib.align(orig, out, task="path", mode="NW")
        nice = edlib.getNiceAlignment(result, orig, out)
        
        aligned_orig = nice["query_aligned"]
        aligned_out = nice["target_aligned"]
        
        # Update max length
        max_len = max(max_len, len(orig))
        
        # Count errors using the aligned sequences
        pos = 0
        for i in range(len(aligned_orig)):
            orig_base = aligned_orig[i]
            out_base = aligned_out[i]
            
            if orig_base == '-':  # Insertion
                total_ins += 1
                pos_errors[pos]['ins'] += 1
            elif out_base == '-':  # Deletion
                total_dels += 1
                pos_errors[pos]['dels'] += 1
                pos += 1
            elif orig_base != out_base:  # Substitution
                total_subs += 1
                pos_errors[pos]['subs'] += 1
                sub_matrix[orig_base][out_base] += 1
                pos += 1
            else:  # Match
                pos += 1
            
            if orig_base != '-':
                pos_errors[pos-1]['total'] += 1

    # Calculate overall error statistics
    total_errors = total_subs + total_ins + total_dels
    total_error_rate = total_errors / total_bases if total_bases > 0 else 0
    
    error_percentages = {
        'substitutions': (total_subs/total_errors*100 if total_errors > 0 else 0),
        'insertions': (total_ins/total_errors*100 if total_errors > 0 else 0),
        'deletions': (total_dels/total_errors*100 if total_errors > 0 else 0)
    }

    # Calculate position-wise error rates
    pos_rates = {pos: {
        'subs_rate': errors['subs']/errors['total'] if errors['total'] > 0 else 0,
        'ins_rate': errors['ins']/errors['total'] if errors['total'] > 0 else 0,
        'del_rate': errors['dels']/errors['total'] if errors['total'] > 0 else 0
    } for pos, errors in pos_errors.items()}

    # Calculate substitution rate matrix
    bases = ['A', 'C', 'G', 'T']
    sub_rates = pd.DataFrame(0.0, index=bases, columns=bases)
    for orig in bases:
        total = sum(sub_matrix[orig].values())
        if total > 0:
            for new in bases:
                sub_rates.loc[orig, new] = float(sub_matrix[orig][new]) / total

    # Calculate k-mer rates
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


def plot_error_analysis(results: Dict, save_path: str = None):
    """Plot comprehensive error analysis"""
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Pie chart of error types
    plt.subplot(3, 2, 1)
    plt.pie(results['error_percentages'].values(), 
            labels=[x.capitalize() for x in results['error_percentages'].keys()],
            autopct='%1.1f%%')
    plt.title('Distribution of Error Types')

    # 2. Position-wise error rates
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

    # 3. Substitution matrix heatmap
    plt.subplot(3, 2, 3)
    sns.heatmap(results['substitution_rates'], 
                annot=True, 
                fmt='.2f', 
                cmap='Blues')
    plt.title('Substitution Rate Matrix')
    
    # 4. K-mer counts comparison
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
    
    # 5. Total error rate
    plt.subplot(3, 2, 5)
    plt.bar(['Total Error Rate'], [results['total_error_rate']])
    plt.ylabel('Rate')
    plt.title('Overall Error Rate')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def parse_cigar(cigar: str):
    """Parse edlib CIGAR string which includes numbers before operations."""
    return [(int(count), op) for count, op in re.findall(r'(\d+)([=XID])', cigar)]


def calculate_position_wise_error_rates(origs: List[str], comparison_seqs: List[str], 
                                      max_length: int) -> Dict[str, np.ndarray]:
    """Calculate position-wise error rates for insertion, deletion, and substitution."""
    n_seqs = len(origs)
    
    # Initialize counters for each error type using max_length
    insertions = np.zeros(max_length)
    deletions = np.zeros(max_length)
    substitutions = np.zeros(max_length)
    valid_positions = np.zeros(max_length)
    
    for orig, comp_seq in zip(origs, comparison_seqs):
        # Update valid positions count
        for i in range(len(orig)):
            valid_positions[i] += 1
            
        # Get alignment
        alignment = edlib.align(orig, comp_seq, task="path", mode="global")
        
        # Process CIGAR string
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
    
    # Convert counts to percentages
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


def plot_position_wise_error_rates(origs: List[str], reads: List[str], outputs: List[str], 
                                 save_path: str = "error_rates.pdf"):
    """Generate and save error rate plots comparing reads and outputs against originals."""
    # Find the maximum length across all sequences
    max_length = max(
        max(len(seq) for seq in origs),
        max(len(seq) for seq in reads),
        max(len(seq) for seq in outputs)
    )
    
    # Calculate error rates for both comparisons using the same max_length
    read_error_rates = calculate_position_wise_error_rates(origs, reads, max_length)
    output_error_rates = calculate_position_wise_error_rates(origs, outputs, max_length)
    
    # Create x-axis range
    x = np.arange(max_length)
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot insertion rates
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
    
    # Plot substitution rates
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
    
    # Plot deletion rates
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
    
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    
    plt.show()


def evaluate_model_predictions(model, data_loader, device, model_type='autoregressive'):
    """Evaluate model predictions and return sequences for analysis"""
    model.eval()
    
    original_seqs = []
    read_seqs = []
    output_seqs = []
    
    with torch.no_grad():
        for src, tgt in data_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            if model_type == 'autoregressive':
                # Generate output sequences
                if hasattr(model, 'module'):
                    generated = model.module.generate(src, max_len=147, temperature=1.0)
                else:
                    generated = model.generate(src, max_len=147, temperature=1.0)
                
                # Decode sequences
                for i in range(src.size(0)):
                    read_seq = decode_one_hot_sequence(src[i], index_to_char)
                    original_seq = decode_one_hot_sequence(tgt[i], index_to_char)
                    output_seq = decode_one_hot_sequence(generated[i], index_to_char)
                    
                    # Clean sequences
                    read_seq = read_seq.replace('S', '').replace('E', '').replace('P', '')
                    original_seq = original_seq.replace('S', '').replace('E', '').replace('P', '')
                    output_seq = output_seq.replace('S', '').replace('E', '').replace('P', '')
                    
                    read_seqs.append(read_seq)
                    original_seqs.append(original_seq)
                    output_seqs.append(output_seq)
            
            # Add VAE evaluation logic here if needed
            
    return original_seqs, read_seqs, output_seqs
