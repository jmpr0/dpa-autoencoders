import pickle
import sys
from ast import literal_eval

import numpy as np
import pandas as pd
import scipy
from common_lib import *
from lz4.frame import open
from scapy.all import *
from statsmodels import robust
from tqdm import tqdm


# check_biflows reads the pcap relating to each single attack of the dataset and groups the individual packets in biflows
def check_biflows(attack_name, basedir='.', verbose=True):
    all_biflows = dict()
    try:
        packets_pcap = PcapReader(
            '%s/Dataset/' % basedir + str(attack_name) + '/' + str(attack_name) + '_pcap.pcapng_sorted.pcap')
    except:
        packets_pcap = PcapReader(
            '%s/Dataset/' % basedir + str(attack_name) + '/' + str(attack_name) + '_pcap_sorted.pcap_sorted.pcap')

    packets_labels = pd.read_csv('%s/Dataset/' % basedir + str(attack_name) + '/' + str(attack_name) + '_labels.csv',
                                 delimiter=',')
    try:
        packets_labels = packets_labels.drop(columns='Unnamed: 0')
    except:
        pass
    packets_labels = packets_labels.to_numpy()
    for i, packet in enumerate(packets_pcap) if not verbose else enumerate(tqdm(packets_pcap)):
        if packet.haslayer(IP) and (packet.haslayer(TCP) or packet.haslayer(UDP)):
            if not (packet.haslayer(NTP) or packet.haslayer(DNS) or packet.sport == 5353 or packet.dport == 5353 or
                    packet[IP].src == '0.0.0.0'):
                src_ip = packet[IP].src
                dst_ip = packet[IP].dst
                src_port = packet.sport
                dst_port = packet.dport

                src = (src_ip, src_port)
                dst = (dst_ip, dst_port)
                proto = packet[IP].proto

                quintupla = (src, dst, proto)
                inverse_quintupla = (dst, src, proto)

                if quintupla not in all_biflows and inverse_quintupla not in all_biflows:
                    all_biflows[quintupla] = []
                    all_biflows[quintupla].append([packet, packets_labels[i][0]])

                elif quintupla in all_biflows:
                    all_biflows[quintupla].append([packet, packets_labels[i][0]])

                elif inverse_quintupla in all_biflows:
                    all_biflows[inverse_quintupla].append([packet, packets_labels[i][0]])

    if not os.path.exists('%s/Biflussi/' % basedir + str(attack_name)):
        os.makedirs('%s/Biflussi/' % basedir + str(attack_name))
    pickle.dump(all_biflows,
                open('%s/Biflussi/' % basedir + str(attack_name) + '/all_biflows_' + str(attack_name) + '.p', 'wb'))


def sort_biflow_by_ts(pkt_list):
    sorted_index = np.argsort([pkt[0].time for pkt in pkt_list])
    return [pkt_list[i] for i in sorted_index]


def get_l4_paylaod_size(pkt):
    """
    To apply when pcap load are truncated.
    :param pkt:
    :return:
    """
    if pkt.haslayer(TCP):
        IP_len = pkt[IP].len
        IP_hdr = pkt[IP].ihl * 4
        TCP_hdr = pkt[TCP].dataofs * 4
        return IP_len - IP_hdr - TCP_hdr
    elif pkt.haslayer(UDP):
        UDP_len = pkt[UDP].len
        return UDP_len - 8
    return 0


# biflow_stats_extraction extracts the statistics of each bi-flow.
# It later distinguishes the malevolent biflows from the benign ones and finally also rescues the biflows that have
# both benign and malevolent packets
def biflow_stats_extraction(attack_name, basedir='.', verbose=True):
    all_biflows = pickle.load(
        open('%s/Biflussi/' % basedir + str(attack_name) + '/all_biflows_' + str(attack_name) + '.p', 'rb'))

    traffic_biflows = dict()

    for quintupla in all_biflows if not verbose else tqdm(all_biflows):
        traffic_biflows[quintupla] = {
            'ttl': [],
            'TCP_Window': [],
            'TCP_flags': [],
            'timestamp': [],
            'pkt_length': [],
            'iat': [],
            'l4_payload': [],
            'pay_length': [],
            'malign': []
        }
        cons_pkt = []

        all_biflows[quintupla] = sort_biflow_by_ts(all_biflows[quintupla])

        for pkt in all_biflows[quintupla] if not verbose else tqdm(all_biflows[quintupla]):
            timestamp = pkt[0].time
            packet_length = pkt[0][IP].len
            time_tl = pkt[0].ttl

            if pkt[0].haslayer(TCP):
                window = pkt[0][TCP].window
                flags = str(pkt[0][TCP].flags)
            else:
                window = 0
                flags = ''

            if pkt[0].haslayer(Raw):
                payload_data = pkt[0].getlayer(Raw).load
            else:
                payload_data = ''
            payload_length = get_l4_paylaod_size(pkt[0])

            cons_pkt.append(pkt[0].time)
            if len(cons_pkt) < 2:
                interarrival_time = 0
            else:
                interarrival_time = np.diff(cons_pkt)[0]
                cons_pkt = cons_pkt[1:]

            malevolo = pkt[1]

            traffic_biflows[quintupla]['ttl'].append(time_tl)
            traffic_biflows[quintupla]['TCP_Window'].append(window)
            traffic_biflows[quintupla]['TCP_flags'].append(flags)
            traffic_biflows[quintupla]['timestamp'].append(timestamp)
            traffic_biflows[quintupla]['pkt_length'].append(packet_length)
            traffic_biflows[quintupla]['iat'].append(interarrival_time)
            traffic_biflows[quintupla]['l4_payload'].append(payload_data)
            traffic_biflows[quintupla]['pay_length'].append(payload_length)
            traffic_biflows[quintupla]['malign'].append(malevolo)

    benign_biflows = dict()
    malign_biflows = dict()
    mix_biflows = dict()
    for quintupla in traffic_biflows if not verbose else tqdm(traffic_biflows):
        benign_biflows[quintupla] = {
            'ttl': [],
            'TCP_Window': [],
            'TCP_flags': [],
            'timestamp': [],
            'pkt_length': [],
            'iat': [],
            'l4_payload': [],
            'pay_length': [],
        }
        malign_biflows[quintupla] = {
            'ttl': [],
            'TCP_Window': [],
            'TCP_flags': [],
            'timestamp': [],
            'pkt_length': [],
            'iat': [],
            'l4_payload': [],
            'pay_length': [],
        }
        for i in range(np.size(traffic_biflows[quintupla]['malign'])):
            if traffic_biflows[quintupla]['malign'][i] == 0:
                benign_biflows[quintupla]['ttl'].append(traffic_biflows[quintupla]['ttl'][i])
                benign_biflows[quintupla]['TCP_Window'].append(traffic_biflows[quintupla]['TCP_Window'][i])
                benign_biflows[quintupla]['TCP_flags'].append(traffic_biflows[quintupla]['TCP_flags'][i])
                benign_biflows[quintupla]['timestamp'].append(traffic_biflows[quintupla]['timestamp'][i])
                benign_biflows[quintupla]['pkt_length'].append(traffic_biflows[quintupla]['pkt_length'][i])
                benign_biflows[quintupla]['iat'].append(traffic_biflows[quintupla]['iat'][i])
                benign_biflows[quintupla]['l4_payload'].append(traffic_biflows[quintupla]['l4_payload'][i])
                benign_biflows[quintupla]['pay_length'].append(traffic_biflows[quintupla]['pay_length'][i])
            else:
                malign_biflows[quintupla]['ttl'].append(traffic_biflows[quintupla]['ttl'][i])
                malign_biflows[quintupla]['TCP_Window'].append(traffic_biflows[quintupla]['TCP_Window'][i])
                malign_biflows[quintupla]['TCP_flags'].append(traffic_biflows[quintupla]['TCP_flags'][i])
                malign_biflows[quintupla]['timestamp'].append(traffic_biflows[quintupla]['timestamp'][i])
                malign_biflows[quintupla]['pkt_length'].append(traffic_biflows[quintupla]['pkt_length'][i])
                malign_biflows[quintupla]['iat'].append(traffic_biflows[quintupla]['iat'][i])
                malign_biflows[quintupla]['l4_payload'].append(traffic_biflows[quintupla]['l4_payload'][i])
                malign_biflows[quintupla]['pay_length'].append(traffic_biflows[quintupla]['pay_length'][i])
        if np.size(benign_biflows[quintupla]['pkt_length']) > 0 and np.size(
                malign_biflows[quintupla]['pkt_length']) > 0:
            mix_biflows[quintupla] = {
                'ttl': [],
                'TCP_Window': [],
                'TCP_flags': [],
                'timestamp': [],
                'pkt_length': [],
                'iat': [],
                'l4_payload': [],
                'pay_length': [],
                'malign': []
            }
            for i in range(np.size(traffic_biflows[quintupla]['malign'])):
                mix_biflows[quintupla]['ttl'].append(traffic_biflows[quintupla]['ttl'][i])
                mix_biflows[quintupla]['TCP_Window'].append(traffic_biflows[quintupla]['TCP_Window'][i])
                mix_biflows[quintupla]['TCP_flags'].append(traffic_biflows[quintupla]['TCP_flags'][i])
                mix_biflows[quintupla]['timestamp'].append(traffic_biflows[quintupla]['timestamp'][i])
                mix_biflows[quintupla]['pkt_length'].append(traffic_biflows[quintupla]['pkt_length'][i])
                mix_biflows[quintupla]['iat'].append(traffic_biflows[quintupla]['iat'][i])
                mix_biflows[quintupla]['l4_payload'].append(traffic_biflows[quintupla]['l4_payload'][i])
                mix_biflows[quintupla]['pay_length'].append(traffic_biflows[quintupla]['pay_length'][i])
                mix_biflows[quintupla]['malign'].append(traffic_biflows[quintupla]['malign'][i])

        if np.size(benign_biflows[quintupla]['pkt_length']) == 0:
            del benign_biflows[quintupla]
        if np.size(malign_biflows[quintupla]['pkt_length']) == 0:
            del malign_biflows[quintupla]

    pickle.dump(mix_biflows,
                open('%s/Biflussi/' % basedir + str(attack_name) + '/mix_biflows_' + str(attack_name) + '.p', 'wb'))
    pickle.dump(benign_biflows,
                open('%s/Biflussi/' % basedir + str(attack_name) + '/benign_biflows_' + str(attack_name) + '.p', 'wb'))
    pickle.dump(malign_biflows,
                open('%s/Biflussi/' % basedir + str(attack_name) + '/malign_biflows_' + str(attack_name) + '.p', 'wb'))


# mix_biflows_splitting divides mixed biflows into benign or malevolent biflows via a filter
def mix_biflows_splitting(attack_name, basedir='.', verbose=True):
    c = pickle.load(open('%s/Biflussi/' % basedir + str(attack_name) + "/mix_biflows_" + str(attack_name) + ".p", "rb"))
    mix = dict()
    for quintupla in c if not verbose else tqdm(c):
        mix[quintupla] = {
            'pkt_benign': 0,
            'pkt_malign': 0
        }

        for i in range(len(c[quintupla]['malign'])):
            if c[quintupla]['malign'][i] == 0:
                mix[quintupla]['pkt_benign'] += 1
            else:
                mix[quintupla]['pkt_malign'] += 1

    mix_buoni = dict()
    mix_malign = dict()

    for quintupla in mix if not verbose else tqdm(mix):
        mix_buoni[quintupla] = {
            'ttl': [],
            'TCP_Window': [],
            'TCP_flags': [],
            'timestamp': [],
            'pkt_length': [],
            'iat': [],
            'l4_payload': [],
            'pay_length': [],
            'malign': []
        }
        mix_malign[quintupla] = {
            'ttl': [],
            'TCP_Window': [],
            'TCP_flags': [],
            'timestamp': [],
            'pkt_length': [],
            'iat': [],
            'l4_payload': [],
            'pay_length': [],
            'malign': []
        }

        for i in range(np.size(c[quintupla]['malign'])) if not verbose else tqdm(
                range(np.size(c[quintupla]['malign']))):
            if (mix[quintupla]['pkt_malign'] == 1 and mix[quintupla]['pkt_benign'] < 20) or (
                    mix[quintupla]['pkt_malign'] / mix[quintupla]['pkt_benign'] < 0.01):
                mix_buoni[quintupla]['ttl'].append(c[quintupla]['ttl'][i])
                mix_buoni[quintupla]['TCP_Window'].append(c[quintupla]['TCP_Window'][i])
                mix_buoni[quintupla]['TCP_flags'].append(c[quintupla]['TCP_flags'][i])
                mix_buoni[quintupla]['timestamp'].append(c[quintupla]['timestamp'][i])
                mix_buoni[quintupla]['pkt_length'].append(c[quintupla]['pkt_length'][i])
                mix_buoni[quintupla]['iat'].append(c[quintupla]['iat'][i])
                mix_buoni[quintupla]['l4_payload'].append(c[quintupla]['l4_payload'][i])
                mix_buoni[quintupla]['pay_length'].append(c[quintupla]['pay_length'][i])
            else:
                mix_malign[quintupla]['ttl'].append(c[quintupla]['ttl'][i])
                mix_malign[quintupla]['TCP_Window'].append(c[quintupla]['TCP_Window'][i])
                mix_malign[quintupla]['TCP_flags'].append(c[quintupla]['TCP_flags'][i])
                mix_malign[quintupla]['timestamp'].append(c[quintupla]['timestamp'][i])
                mix_malign[quintupla]['pkt_length'].append(c[quintupla]['pkt_length'][i])
                mix_malign[quintupla]['iat'].append(c[quintupla]['iat'][i])
                mix_malign[quintupla]['l4_payload'].append(c[quintupla]['l4_payload'][i])
                mix_malign[quintupla]['pay_length'].append(c[quintupla]['pay_length'][i])

        if np.size(mix_buoni[quintupla]['timestamp']) == 0:
            del mix_buoni[quintupla]
        if np.size(mix_malign[quintupla]['timestamp']) == 0:
            del mix_malign[quintupla]

    pickle.dump(mix_buoni,
                open('%s/Biflussi/' % basedir + str(attack_name) + '/mix_benign_biflows_' + str(attack_name) + '.p',
                     'wb'))
    pickle.dump(mix_malign,
                open('%s/Biflussi/' % basedir + str(attack_name) + '/mix_malign_biflows_' + str(attack_name) + '.p',
                     'wb'))


def load_biflows(attack_name, basedir='.'):
    benign_biflows = pickle.load(
        open('%s/Biflussi/' % basedir + str(attack_name) + "/benign_biflows_" + str(attack_name) + ".p", "rb"))
    malign_biflows = pickle.load(
        open('%s/Biflussi/' % basedir + str(attack_name) + "/malign_biflows_" + str(attack_name) + ".p", "rb"))
    mix_biflows = pickle.load(
        open('%s/Biflussi/' % basedir + str(attack_name) + "/mix_biflows_" + str(attack_name) + ".p", "rb"))
    mix_b_biflows = pickle.load(
        open('%s/Biflussi/' % basedir + str(attack_name) + "/mix_benign_biflows_" + str(attack_name) + ".p", "rb"))
    mix_m_biflows = pickle.load(
        open('%s/Biflussi/' % basedir + str(attack_name) + "/mix_malign_biflows_" + str(attack_name) + ".p", "rb"))
    return benign_biflows, malign_biflows, mix_biflows, mix_b_biflows, mix_m_biflows


# add_filtering takes the mixed good and the bad mixed and adds them to the previously saved good and bad biflows
def add_filtering(attack_name, basedir='.', verbose=True):
    a, b, c, d, e = load_biflows(attack_name, basedir)

    for quintupla in c if not verbose else tqdm(c):
        del a[quintupla]
        del b[quintupla]
    for quintupla in d if not verbose else tqdm(d):
        a[quintupla] = d[quintupla]

    for quintupla in e if not verbose else tqdm(e):
        b[quintupla] = e[quintupla]

    pickle.dump(a,
                open('%s/Biflussi/' % basedir + str(attack_name) + '/final_benign_biflows_' + str(attack_name) + '.p',
                     'wb'))
    pickle.dump(b,
                open('%s/Biflussi/' % basedir + str(attack_name) + '/final_malign_biflows_' + str(attack_name) + '.p',
                     'wb'))


# extract_statistics computes various statistics related to the characteristics of each single biflow
def extract_statistics(packettino, dropfirst=False):
    packet = list(map(float, packettino))
    if dropfirst:
        packet = packet[1:]

    try:
        packet_field_statistics = dict()
        packet_field_statistics['min'] = np.min(packet)
        packet_field_statistics['max'] = np.max(packet)
        packet_field_statistics['mean'] = np.mean(packet)
        packet_field_statistics['std'] = np.std(packet)
        packet_field_statistics['var'] = np.var(packet)
        packet_field_statistics['mad'] = robust.mad(packet, c=1)
        packet_field_statistics['skew'] = scipy.stats.skew(packet, bias=False)
        packet_field_statistics['kurtosis'] = scipy.stats.kurtosis(packet, bias=False)
        packet_field_statistics['10_percentile'] = np.percentile(packet, 10)
        packet_field_statistics['20_percentile'] = np.percentile(packet, 20)
        packet_field_statistics['30_percentile'] = np.percentile(packet, 30)
        packet_field_statistics['40_percentile'] = np.percentile(packet, 40)
        packet_field_statistics['50_percentile'] = np.percentile(packet, 50)
        packet_field_statistics['60_percentile'] = np.percentile(packet, 60)
        packet_field_statistics['70_percentile'] = np.percentile(packet, 70)
        packet_field_statistics['80_percentile'] = np.percentile(packet, 80)
        packet_field_statistics['90_percentile'] = np.percentile(packet, 90)

    except ValueError:
        packet_field_statistics = dict()
        packet_field_statistics['min'] = np.nan
        packet_field_statistics['max'] = np.nan
        packet_field_statistics['mean'] = np.nan
        packet_field_statistics['std'] = np.nan
        packet_field_statistics['var'] = np.nan
        packet_field_statistics['mad'] = np.nan
        packet_field_statistics['skew'] = np.nan
        packet_field_statistics['kurtosis'] = np.nan
        packet_field_statistics['10_percentile'] = np.nan
        packet_field_statistics['20_percentile'] = np.nan
        packet_field_statistics['30_percentile'] = np.nan
        packet_field_statistics['40_percentile'] = np.nan
        packet_field_statistics['50_percentile'] = np.nan
        packet_field_statistics['60_percentile'] = np.nan
        packet_field_statistics['70_percentile'] = np.nan
        packet_field_statistics['80_percentile'] = np.nan
        packet_field_statistics['90_percentile'] = np.nan
        sys.stderr.write('WARNING!\n')

    return packet_field_statistics


# ExtractStatisticsBiflow Extracts the statistics of each benign and then malevolent biflows and groups them,
# then saving them all in a csv of the statistics of the benign and then the malevolent biflows
def extract_stats_computing(attack_name, basedir='.', verbose=True, backup=True):
    benign_biflows = pickle.load(
        open('%s/Biflussi/' % basedir + str(attack_name) + '/final_benign_biflows_' + str(attack_name) + '.p', 'rb'))
    malign_biflows = pickle.load(
        open('%s/Biflussi/' % basedir + str(attack_name) + '/final_malign_biflows_' + str(attack_name) + '.p', 'rb'))
    pkt_length_tot_quintupla = 0
    n_pkt_quintupla = 0
    statistics = dict()
    statistiche_biflusso = dict()
    for (pkt, attack_namea) in [(benign_biflows, "benign_biflows"), (malign_biflows, "malign_biflows")]:
        for quintupla in pkt if not verbose else tqdm(pkt):
            statistiche_biflusso[quintupla] = {
                'num_pkt': int,
                'tot_pay_length': float,
                'mean_pay_length': float,
                'std_pay_length': float,
                'max_pay_length': float,
                'min_pay_length': float,
                'mad_pay_length': float,
                'kurtosis_pay_length': float,
                'skew_pay_length': float,
                'var_pay_length': float,
                '10_percentile_pay_length': float,
                '20_percentile_pay_length': float,
                '30_percentile_pay_length': float,
                '40_percentile_pay_length': float,
                '50_percentile_pay_length': float,
                '60_percentile_pay_length': float,
                '70_percentile_pay_length': float,
                '80_percentile_pay_length': float,
                '90_percentile_pay_length': float,
                'mean_iat': float,
                'std_iat': float,
                'max_iat': float,
                'min_iat': float,
                'mad_iat': float,
                'kurtosis_iat': float,
                'skew_iat': float,
                'var_iat': float,
                '10_percentile_iat': float,
                '20_percentile_iat': float,
                '30_percentile_iat': float,
                '40_percentile_iat': float,
                '50_percentile_iat': float,
                '60_percentile_iat': float,
                '70_percentile_iat': float,
                '80_percentile_iat': float,
                '90_percentile_iat': float,
                'byte_rate': float,
                'mean_ttl': float,
                'std_ttl': float,
                'max_ttl': float,
                'min_ttl': float,
                'mad_ttl': float,
                'kurtosis_ttl': float,
                'skew_ttl': float,
                'var_ttl': float,
                '10_percentile_ttl': float,
                '20_percentile_ttl': float,
                '30_percentile_ttl': float,
                '40_percentile_ttl': float,
                '50_percentile_ttl': float,
                '60_percentile_ttl': float,
                '70_percentile_ttl': float,
                '80_percentile_ttl': float,
                '90_percentile_ttl': float,
                'mean_TCP_Window': float,
                'std_TCP_Window': float,
                'max_TCP_Window': float,
                'min_TCP_Window': float,
                'mad_TCP_Window': float,
                'kurtosis_TCP_Window': float,
                'skew_TCP_Window': float,
                'var_TCP_Window': float,
                '10_percentile_TCP_Window': float,
                '20_percentile_TCP_Window': float,
                '30_percentile_TCP_Window': float,
                '40_percentile_TCP_Window': float,
                '50_percentile_TCP_Window': float,
                '60_percentile_TCP_Window': float,
                '70_percentile_TCP_Window': float,
                '80_percentile_TCP_Window': float,
                '90_percentile_TCP_Window': float,
                'count_FIN_TCP_flags': int,
                'count_SYN_TCP_flags': int,
                'count_RST_TCP_flags': int,
                'count_PSH_TCP_flags': int,
                'count_ACK_TCP_flags': int,
                'count_URG_TCP_flags': int,
                'count_ECE_TCP_flags': int,
                'count_CWR_TCP_flags': int,
            }

            statistiche_biflusso[quintupla]['num_pkt'] = len(pkt[quintupla]['timestamp'])
            for i in range(0, len(pkt[quintupla]['timestamp'])):
                pkt_length_tot_quintupla += pkt[quintupla]['pay_length'][i]

            statistiche_biflusso[quintupla]['tot_pay_length'] = pkt_length_tot_quintupla
            statistics[quintupla] = extract_statistics(pkt[quintupla]['pay_length'])
            statistiche_biflusso[quintupla]['mean_pay_length'] = statistics[quintupla]['mean']
            statistiche_biflusso[quintupla]['std_pay_length'] = statistics[quintupla]['std']
            statistiche_biflusso[quintupla]['min_pay_length'] = statistics[quintupla]['min']
            statistiche_biflusso[quintupla]['max_pay_length'] = statistics[quintupla]['max']
            statistiche_biflusso[quintupla]['mad_pay_length'] = statistics[quintupla]['mad']
            statistiche_biflusso[quintupla]['kurtosis_pay_length'] = statistics[quintupla]['kurtosis']
            statistiche_biflusso[quintupla]['skew_pay_length'] = statistics[quintupla]['skew']
            statistiche_biflusso[quintupla]['var_pay_length'] = statistics[quintupla]['var']
            statistiche_biflusso[quintupla]['10_percentile_pay_length'] = statistics[quintupla]['10_percentile']
            statistiche_biflusso[quintupla]['20_percentile_pay_length'] = statistics[quintupla]['20_percentile']
            statistiche_biflusso[quintupla]['30_percentile_pay_length'] = statistics[quintupla]['30_percentile']
            statistiche_biflusso[quintupla]['40_percentile_pay_length'] = statistics[quintupla]['40_percentile']
            statistiche_biflusso[quintupla]['50_percentile_pay_length'] = statistics[quintupla]['50_percentile']
            statistiche_biflusso[quintupla]['60_percentile_pay_length'] = statistics[quintupla]['60_percentile']
            statistiche_biflusso[quintupla]['70_percentile_pay_length'] = statistics[quintupla]['70_percentile']
            statistiche_biflusso[quintupla]['80_percentile_pay_length'] = statistics[quintupla]['80_percentile']
            statistiche_biflusso[quintupla]['90_percentile_pay_length'] = statistics[quintupla]['90_percentile']

            statistics[quintupla] = extract_statistics(pkt[quintupla]['iat'], dropfirst=True)
            statistiche_biflusso[quintupla]['mean_iat'] = statistics[quintupla]['mean']
            statistiche_biflusso[quintupla]['std_iat'] = statistics[quintupla]['std']
            statistiche_biflusso[quintupla]['min_iat'] = statistics[quintupla]['min']
            statistiche_biflusso[quintupla]['max_iat'] = statistics[quintupla]['max']
            statistiche_biflusso[quintupla]['mad_iat'] = statistics[quintupla]['mad']
            statistiche_biflusso[quintupla]['kurtosis_iat'] = statistics[quintupla]['kurtosis']
            statistiche_biflusso[quintupla]['skew_iat'] = statistics[quintupla]['skew']
            statistiche_biflusso[quintupla]['var_iat'] = statistics[quintupla]['var']
            statistiche_biflusso[quintupla]['10_percentile_iat'] = statistics[quintupla]['10_percentile']
            statistiche_biflusso[quintupla]['20_percentile_iat'] = statistics[quintupla]['20_percentile']
            statistiche_biflusso[quintupla]['30_percentile_iat'] = statistics[quintupla]['30_percentile']
            statistiche_biflusso[quintupla]['40_percentile_iat'] = statistics[quintupla]['40_percentile']
            statistiche_biflusso[quintupla]['50_percentile_iat'] = statistics[quintupla]['50_percentile']
            statistiche_biflusso[quintupla]['60_percentile_iat'] = statistics[quintupla]['60_percentile']
            statistiche_biflusso[quintupla]['70_percentile_iat'] = statistics[quintupla]['70_percentile']
            statistiche_biflusso[quintupla]['80_percentile_iat'] = statistics[quintupla]['80_percentile']
            statistiche_biflusso[quintupla]['90_percentile_iat'] = statistics[quintupla]['90_percentile']

            statistics[quintupla] = extract_statistics(pkt[quintupla]['ttl'])
            statistiche_biflusso[quintupla]['mean_ttl'] = statistics[quintupla]['mean']
            statistiche_biflusso[quintupla]['std_ttl'] = statistics[quintupla]['std']
            statistiche_biflusso[quintupla]['min_ttl'] = statistics[quintupla]['min']
            statistiche_biflusso[quintupla]['max_ttl'] = statistics[quintupla]['max']
            statistiche_biflusso[quintupla]['mad_ttl'] = statistics[quintupla]['mad']
            statistiche_biflusso[quintupla]['kurtosis_ttl'] = statistics[quintupla]['kurtosis']
            statistiche_biflusso[quintupla]['skew_ttl'] = statistics[quintupla]['skew']
            statistiche_biflusso[quintupla]['var_ttl'] = statistics[quintupla]['var']
            statistiche_biflusso[quintupla]['10_percentile_ttl'] = statistics[quintupla]['10_percentile']
            statistiche_biflusso[quintupla]['20_percentile_ttl'] = statistics[quintupla]['20_percentile']
            statistiche_biflusso[quintupla]['30_percentile_ttl'] = statistics[quintupla]['30_percentile']
            statistiche_biflusso[quintupla]['40_percentile_ttl'] = statistics[quintupla]['40_percentile']
            statistiche_biflusso[quintupla]['50_percentile_ttl'] = statistics[quintupla]['50_percentile']
            statistiche_biflusso[quintupla]['60_percentile_ttl'] = statistics[quintupla]['60_percentile']
            statistiche_biflusso[quintupla]['70_percentile_ttl'] = statistics[quintupla]['70_percentile']
            statistiche_biflusso[quintupla]['80_percentile_ttl'] = statistics[quintupla]['80_percentile']
            statistiche_biflusso[quintupla]['90_percentile_ttl'] = statistics[quintupla]['90_percentile']

            statistics[quintupla] = extract_statistics(pkt[quintupla]['TCP_Window'])
            statistiche_biflusso[quintupla]['mean_TCP_Window'] = statistics[quintupla]['mean']
            statistiche_biflusso[quintupla]['std_TCP_Window'] = statistics[quintupla]['std']
            statistiche_biflusso[quintupla]['min_TCP_Window'] = statistics[quintupla]['min']
            statistiche_biflusso[quintupla]['max_TCP_Window'] = statistics[quintupla]['max']
            statistiche_biflusso[quintupla]['mad_TCP_Window'] = statistics[quintupla]['mad']
            statistiche_biflusso[quintupla]['kurtosis_TCP_Window'] = statistics[quintupla]['kurtosis']
            statistiche_biflusso[quintupla]['skew_TCP_Window'] = statistics[quintupla]['skew']
            statistiche_biflusso[quintupla]['var_TCP_Window'] = statistics[quintupla]['var']
            statistiche_biflusso[quintupla]['10_percentile_TCP_Window'] = statistics[quintupla]['10_percentile']
            statistiche_biflusso[quintupla]['20_percentile_TCP_Window'] = statistics[quintupla]['20_percentile']
            statistiche_biflusso[quintupla]['30_percentile_TCP_Window'] = statistics[quintupla]['30_percentile']
            statistiche_biflusso[quintupla]['40_percentile_TCP_Window'] = statistics[quintupla]['40_percentile']
            statistiche_biflusso[quintupla]['50_percentile_TCP_Window'] = statistics[quintupla]['50_percentile']
            statistiche_biflusso[quintupla]['60_percentile_TCP_Window'] = statistics[quintupla]['60_percentile']
            statistiche_biflusso[quintupla]['70_percentile_TCP_Window'] = statistics[quintupla]['70_percentile']
            statistiche_biflusso[quintupla]['80_percentile_TCP_Window'] = statistics[quintupla]['80_percentile']
            statistiche_biflusso[quintupla]['90_percentile_TCP_Window'] = statistics[quintupla]['90_percentile']

            for flag in ['FIN', 'SYN', 'RST', 'PSH', 'ACK', 'URG', 'ECE', 'CWR']:
                statistiche_biflusso[quintupla]['count_%s_TCP_flags' % flag] = sum(
                    [flag[0] in v for v in pkt[quintupla]['TCP_flags']]
                )

            if len(pkt[quintupla]['timestamp']) > 1:
                statistiche_biflusso[quintupla]['byte_rate'] = pkt_length_tot_quintupla / (
                        pkt[quintupla]['timestamp'][-1] - pkt[quintupla]['timestamp'][0])
            else:
                statistiche_biflusso[quintupla]['byte_rate'] = 0

            pkt_length_tot_quintupla = 0

        data = pd.DataFrame.from_dict(statistiche_biflusso, orient='index')
        data.columns = data.columns.astype(str)
        if not os.path.exists('%s/Statistiche/' % basedir + str(attack_name)):
            os.makedirs('%s/Statistiche/' % basedir + str(attack_name))
        if backup:
            to_csv_bk(data, '%s/Statistiche/' % basedir + str(attack_name) + '/Statistiche_Biflussi_' + str(
                attack_namea) + '.csv')
        else:
            data.to_csv(
                '%s/Statistiche/' % basedir + str(attack_name) + '/Statistiche_Biflussi_' + str(attack_namea) + '.csv')
        statistiche_biflusso.clear()


def create_train_dataset(basedir='.', backup=True):
    path = "%s/Statistiche/" % basedir
    benign_df = pd.DataFrame()
    benign_label = pd.DataFrame()
    malign_label = pd.DataFrame()
    malign_df = pd.DataFrame()
    for attack_name, num in str_to_int.items():
        temp_benign = pd.read_csv(str(path) + str(attack_name) + '/Statistiche_Biflussi_benign_biflows.csv')
        temp_malign = pd.read_csv(str(path) + str(attack_name) + '/Statistiche_Biflussi_malign_biflows.csv')
        if temp_benign.empty:
            print(str(attack_name), "benign vuoto")
        else:
            temp_benign = temp_benign.drop(['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2'], axis=1)
        benign_labels = np.zeros((len(temp_benign), 2))
        for i in range(len(temp_benign)):
            benign_labels[i][1] = num
        if len(benign_labels) == 0:
            None
        else:
            benign_labels = pd.DataFrame(benign_labels, columns=['Malign', 'Class'])
            benign_df = pd.concat([benign_df, temp_benign])
            benign_label = pd.concat([benign_label, benign_labels])

        temp_malign = temp_malign.drop(['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2'], axis=1)
        malign_labels = np.ones((len(temp_malign), 2))
        for i in range(len(temp_malign)):
            malign_labels[i][1] = num
        malign_labels = pd.DataFrame(malign_labels, columns=['Malign', 'Class'])

        malign_df = pd.concat([malign_df, temp_malign])
        malign_label = pd.concat([malign_label, malign_labels])

    if not os.path.exists(str(path) + "Biflussi_Totali/"):
        os.makedirs(str(path) + "Biflussi_Totali/")

    if backup:
        to_csv_bk(benign_df, str(path) + "Biflussi_Totali/benign_biflows_data.csv", index=False)
        to_csv_bk(benign_label, str(path) + "Biflussi_Totali/benign_biflows_labels.csv", index=False)
        to_csv_bk(malign_df, str(path) + "Biflussi_Totali/malign_biflows_data.csv", index=False)
        to_csv_bk(malign_label, str(path) + "Biflussi_Totali/malign_biflows_labels.csv", index=False)
    else:
        benign_df.to_csv(str(path) + "Biflussi_Totali/benign_biflows_data.csv", index=False)
        benign_label.to_csv(str(path) + "Biflussi_Totali/benign_biflows_labels.csv", index=False)
        malign_df.to_csv(str(path) + "Biflussi_Totali/malign_biflows_data.csv", index=False)
        malign_label.to_csv(str(path) + "Biflussi_Totali/malign_biflows_labels.csv", index=False)


def data_compact(basedir='.', backup=True):
    path = "%s/Statistiche/" % basedir
    x = pd.read_csv("%s/Statistiche/Biflussi_Totali/benign_biflows_data.csv" % basedir)
    x_label = pd.read_csv("%s/Statistiche/Biflussi_Totali/benign_biflows_labels.csv" % basedir)
    x1 = pd.read_csv("%s/Statistiche/Biflussi_Totali/malign_biflows_data.csv" % basedir)
    x1_label = pd.read_csv("%s/Statistiche/Biflussi_Totali/malign_biflows_labels.csv" % basedir)

    complete_x = pd.concat([x, x_label], axis=1)
    complete_x1 = pd.concat([x1, x1_label], axis=1)
    train_data = pd.concat([complete_x, complete_x1], axis=0)

    if not os.path.exists(str(path) + "Biflussi_Totali/"):
        os.makedirs(str(path) + "Biflussi_Totali/")

    train_data.reset_index(inplace=True, drop=True)

    if backup:
        to_csv_bk(train_data, str(path) + "Biflussi_Totali/training_set_all.csv", index=False)
    else:
        train_data.to_csv(str(path) + "Biflussi_Totali/training_set_all.csv", index=False)

    print(len(train_data))

    # Removing duplicates rows w/ labels taking only the first occurrence.
    train_data = train_data.drop_duplicates()
    print(len(train_data))
    # Removing remaining duplicates, which are conflictual labels. We discard all the rows match the filter.
    no_conflicts_index = train_data.iloc[:, :-2].drop_duplicates(keep=False).index
    train_data = train_data.loc[no_conflicts_index, :]
    print(len(train_data))
    # Removing 1-sized rows.
    train_data = train_data[train_data['num_pkt'] != 1]
    print(len(train_data))

    if backup:
        to_csv_bk(train_data, str(path) + "Biflussi_Totali/training_set_nodup.csv", index=False)
    else:
        train_data.to_csv(str(path) + "Biflussi_Totali/training_set_nodup.csv", index=False)


if __name__ == '__main__':

    args_names = ['<BASEDIR>']
    optargs_names = ['<ATTACK_NAMES=[default list in the script]>']

    if len(sys.argv) <= len(args_names):
        print('Usage:', sys.argv[0], ' '.join(args_names), '{', ' '.join(optargs_names), '}')
        exit()

    basedir = sys.argv[1]
    attack_names = literal_eval(sys.argv[2]) if len(sys.argv) > 2 else [
        "Active Wiretap", "ARP MitM", "Fuzzing", "OS Scan", "SSDP Flood",
        "SSL Renegotiation", "SYN DoS", "Video Injection", "Mirai"
    ]

    for attack_name in attack_names:
        print('Elaborating attack', attack_name)
        check_biflows(attack_name, basedir)
        biflow_stats_extraction(attack_name, basedir)
        mix_biflows_splitting(attack_name, basedir)
        add_filtering(attack_name, basedir)
        extract_stats_computing(attack_name, basedir)
    create_train_dataset(basedir)
    data_compact(basedir)
    print('DONE!')
