"""Using a simplistic channel model, compute sum data rate over multiple timeslots for multiple terminals assigned to multiple basestations. Supports multiple combining models. 

Example number for typical LTE numeroloy ; https://sites.google.com/site/lteencyclopedia/lte-radio-link-budgeting-and-rf-planning 
"""


import numpy as np 


num_timeslots = 2  # before we periodically repeat schedules 
timeslot_length = 0.01 # in seconds

shannon_discount_dB = 0.5 # in dB
shannon_discount = 10**(shannon_discount_dB/10)

bandwidth = 9*10**6 # bandwidth for an LTE setup 
noise_dBm = -95  # for a 9MHz channel , comprising thermal and device noise. this is total noise floor  
noise_mW = 10**(noise_dBm/10)

frequency = 2*10**9 # main carrier, in Hz  

transmit_power_dBm = 40 # no antenna gain, but some losses 
antenna_gain_dB = 0
cable_loss_dB = 0
eirp_dBm = transmit_power_dBm + antenna_gain_dB + cable_loss_dB



### EXAMPLE PARAMETERS

num_basestations = 2
num_ues = 4
playground_size = 10000 
num_iterations = 100


def data_rate_factory(type):
    """map SINR to data rate"""

    def discounted_shannon(sinr_db,
                               bandwidth=bandwidth,
                               shannon_discount_dB=shannon_discount_dB):
        discounted_sinr = 10**((sinr_db - shannon_discount_dB)/10)
        return bandwidth * np.log10(1 + discounted_sinr)

    if type=="discounted_shannon":
        return discounted_shannon
    else:
        raise NotImplemented
    
    
def path_loss_factory(type):
    """path loss in db"""
    def path_loss_okumura_hata(distance):
        """distance in km"""
        return const1 + const2*np.log10(distance) 

    if type=="suburban_indoor":
        hb = 50 # base station height in meter
        f = frequency/1000000 # we need frequency in MHz here 
        hm = 1.5 # mobile height in meter

        CH = 0.8 + (1.1*np.log10(f) - 0.7)*hm - 1.56*np.log10(f) 
    else:
        raise NotImplemented 
    
    const1 = 69.55 + 26.16*np.log10(f) - 13.82*np.log10(hb) - CH
    const2 = 44.9 - 6.55 * np.log10(hb)
    return path_loss_okumura_hata


def data_volume(data_rate, timeslot_length=timeslot_length):
    return data_rate*timeslot_length 

def received_power(path_loss_db, eirp_dBm=eirp_dBm):
    return 10**((eirp_dBm -path_loss_db)/10)

##########################################

def compute_sum_datavolume(distances, downlink_schedule, uplink_schedule,
                          path_loss_fct, data_rate_fct,
                          num_timeslots=num_timeslots,
                          num_bs=num_basestations, num_ue=num_ues):
    """compute downlink and uplink volume for the given distances and schedules; 
    for one scheduling round of of num_timeslots many slots. 
    So far, no combining; just sum up over all timeslots.  

    Note: Some combining schemes can combine SNRs across timeslots, 
    so we cannot simply do this per timeslot, indepedently.
    """

    # downlink, first: how much does each UE receive?
    # NOTE: horrible code; needs to be rewritten for proper Numpy indexing
    downlink_volume = [0] * num_ues
    for schedule in downlink_schedule:
        served = [False] * num_ues
        this_slot_downlink_rate = [0] * num_ues
        bs_activity = np.sum(schedule, axis=1)
        # print("Schedule: ", schedule, bs_activity)

        # I am sure the following can be done much nicer in numpy: 
        sending_bs = [i for i, a in enumerate(bs_activity) if a == 1]
        silly_bs = [i for i, a in enumerate(bs_activity) if a >1]
        silent_bs = [i for i, a in enumerate(bs_activity) if a == 0]
        # print("Sending: ", sending_bs, "Silent", silent_bs, "Silly: ", silly_bs)
        for ue in range(num_ues):
            for bs in sending_bs:
                # print(f"UE {ue},  BS {bs}" )
                if schedule[bs, ue] == 1:
                    # who interferes?
                    interfering_bs = set(sending_bs) - {bs}

                    signal_distance = distances[bs][ue]
                    signal_path_loss = path_loss_fct(signal_distance)
                    signal = received_power(signal_path_loss)
                    
                    interference = sum(received_power(path_loss_fct(distances[ibs][ue]))
                                        for ibs in interfering_bs)

                    sinr = signal / (noise_mW +  interference)
                    volume = data_volume(data_rate_fct(10*np.log10(sinr)))
                    downlink_volume[ue] += volume
                    # print (f"signal distance {signal_distance}",
                    #         f"signal path loss {signal_path_loss}", signal, interference, sinr, volume) 

    return downlink_volume 

##########################################

def random_schedules(num_basestatins, num_ues ):

# Randomly choose some schedules
# downlink: from BS to UE, so BS in rows, UEs in columns 
    downlink_schedule = [
        np.random.binomial(1, 0.3, (num_basestations, num_ues))
        for t in range(num_timeslots)]
    # uplink: vice versa 
    uplink_schedule = [
        np.random.binomial(1, 0.1, (num_ues, num_basestations))
        for t in range(num_timeslots)]

    return downlink_schedule, uplink_schedule 
            
            
            
def get_setup(type): 
    def random_setup():
        basestation_locations = np.random.uniform(0, playground_size, (num_basestations, 2))
        ue_locations = np.random.uniform(0, playground_size, (num_ues, 2))

        downlink_schedule, uplink_schedule = random_schedules(num_basestations, num_ues)
        
        return basestation_locations, ue_locations, downlink_schedule, uplink_schedule 
            
            
    def simple_setup():

        basestation_locations = np.array([ [0, 0], [1, 0 ]  ])
        ue_locations = np.array([[.1, .1 ], [.1, .9], [.9, .1 ], [.9, .9 ],  ])

        downlink_schedule = [
            np.array([ [1, 0, 0, 0], [0, 1, 0, 0 ] ]), # TS 1 
            np.array([ [0, 0, 1, 0 ], [0, 0, 0, 1] ]), # TS 2 
            ]
        uplink_schedule = None
        return basestation_locations, ue_locations, downlink_schedule, uplink_schedule 
        
    
    if type=="random":
        return random_setup()
    elif type=="simple":
        return simple_setup()
    else:
        raise NotImplemented 


def get_distances (basestation_locations, ue_locations):
    # this is brute force; surely must be done more efficiently for large simulations; only compute
    # distances that are actually needed; more tuning needed 
    # compare https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
    # from BS in rows to UEs in columns: 
    distances = [
        np.sqrt(np.sum( (ue_locations - bs)**2, axis=1))
        for bs in basestation_locations]
    
    return distances 
    
def main():
    pl = path_loss_factory("suburban_indoor")
    dr = data_rate_factory("discounted_shannon")

    
    # quick and dirty sanity check: 
    # for d in [0.001, 0.01, .1, .141, 1, 1.5, 2, 5, 10, 141.42]:
    #     pld = pl(d)
    #     rx_dBm = eirp_dBm - pld
    #     rx = 10**(rx_dBm/10)
    #     snr_dB = rx_dBm - noise_dBm
    #     snr = 10**(snr_dB/10)
    #     print (d, pld, rx_dBm, rx, snr_dB, snr)

    # print("====================")



    # setup example input 

    basestation_locations, ue_locations, downlink_schedule, uplink_schedule = get_setup("simple")

    distances = get_distances(basestation_locations, ue_locations)
    
    print("Distances:")
    print(distances) 


    # print("DL:")
    # print(downlink_schedule)
    # print("UL:")
    # print(uplink_schedule)

    ###
    rtotal = np.zeros(num_ues)
        
    for i in range(num_iterations):
        r = compute_sum_datavolume(distances,
                                    downlink_schedule, uplink_schedule,
                                    pl, dr)
        rtotal += np.array(r)
        downlink_schedule, uplink_schedule = random_schedules(
            num_basestations, num_ues)

    print("RESULT:")
    print(r)
    print("RATES (in Mbit/s):")
    print([x/(num_timeslots*timeslot_length)/1024**2/num_iterations for x in r])
    
if __name__ == '__main__':
    main()

