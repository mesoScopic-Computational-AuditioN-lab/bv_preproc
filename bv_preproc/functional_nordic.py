"""Functions for doing nordic from brainvoyager, functions 
functions created by Mahdi Enan (2022)"""

## TO BE IMPLEMENTED!
## NOT TESTED




## HIGH LEVEL FUNCTIONS (LOOPIN OVER PARTICIPANTS AND SESSIONS




class BVPP():

    def __init__(self, bv):
        self.bv = bv

    

    # nordic specific stuff
    def extract_files_gzip(self, filelist):
        new_filenames = []
        for file in filelist:
            extracted_file = file.replace('.gz', '')
            new_filenames.append(extracted_file)
            with gzip.open(file, 'rb') as f_in:
                with open(extracted_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        return new_filenames

    def run_nordic(self, condition_path):
        os.chdir(condition_path)
        cmd = ['matlab', '-nodisplay', '-nosplash', '-r', "run('RunNordic_Github_v2.m');exit;"]
        pprint(subprocess.check_output(cmd))

    # prt specific routines

    def create_log_dict(self, log_path, filename):

        with open(log_path + filename, 'r') as f:
            log = [np.array(list(filter(None, l.strip('\n').split('\t')))) for l in f.readlines()]
        
        keys = log[0]
        values = log[2:]

        log_data = {k:[] for k in keys}
        log_data['vol'] = []

        datetime_shift = datetime.strptime(values[0][-2], '%H:%M:%S.%f')

        for line in values:
            # discard test stimuli
            if line[0] == 'x': continue

            if len(line) == 7:
                trial, event, stim, start_time, duration, relative, volume = line
                exp = ''
            elif len(line) == 8:
                trial, event, stim, exp, start_time, duration, relative, volume = line

            log_data['Trial'].append(trial)
            log_data['Event'].append(event)
            log_data['Stim'].append(stim)
            log_data['Exp'].append(exp)
            log_data['Start Time'].append(start_time)
            log_data['Duration'].append(duration)
            relative_datetime = datetime.strptime(relative, '%H:%M:%S.%f')

            fixed_datetime = relative_datetime - datetime_shift
            fixed_datetime= float(f'{fixed_datetime.seconds}.{fixed_datetime.microseconds}')
            log_data['Relative'].append(fixed_datetime)

            log_data['vol'].append(int(fixed_datetime//2.5) + 1)

        return log_data

    def create_prt_data(self, log_data, conditions):
        sound_events = np.where(np.array(log_data['Event']) == 'Sound')[0]

        prt_data = {condition:[] for condition in conditions}

        for sound_idx in sound_events:
            sound = log_data["Stim"][sound_idx]
            exp = log_data["Exp"][sound_idx-1]

            # WhatOff
            if 'random' in log_data['Stim'][sound_idx-1]:
                condition = 'WhatOff_'
                if sound == 'omis':
                    condition += 'Omission'
                elif sound == 'pa':
                    condition += 'PA'
                elif sound == 'ga':
                    condition += 'GA'

            else:
                condition = 'WhatOn_'
                if sound == exp == 'pa':
                    condition += 'Congruent_PA'
                elif sound == exp == 'ga':
                    condition += 'Congruent_GA'
                elif sound == 'pa' and exp == 'ga':
                    condition += 'Incongruent_GA'
                elif sound == 'ga' and exp == 'pa':
                    condition += 'Incongruent_PA'
                elif sound == 'omis' and exp == 'pa':
                    condition += 'Omission_PA'
                elif sound == 'omis' and exp == 'ga':
                    condition += 'Omission_GA'

            end = log_data['vol'][sound_idx]
            
            # find start
            start_idx = sound_idx-1
            while log_data['Stim'][start_idx] != 'dummy':
                start_idx -= 1
            start = log_data['vol'][start_idx]

            trial = log_data['Trial'][sound_idx]
            # print(f'trial: {trial} \t expected: {exp} \t sound: {sound} \t condition: {condition}\t\t start: {start}\t\t end: {end}')
            prt_data[condition].append({'trial': trial, 'start': start, 'end': end})

        # add motor response entries
        volumes = np.array(log_data['vol'])[np.where((np.array(log_data['Event']) == 'Action') & 
                                                ((np.array(log_data['Stim']) == 'ga') | 
                                                 (np.array(log_data['Stim']) == 'pa'))
                                               )]
        trials = np.array(log_data['Trial'])[np.where((np.array(log_data['Event']) == 'Action') & 
            ((np.array(log_data['Stim']) == 'ga') | 
            (np.array(log_data['Stim']) == 'pa')))]

        for vol_idx, vol in enumerate(volumes):
            prt_data['MotorResponse'].append({'trial': trials[vol_idx], 'start': vol, 'end': vol})

        return prt_data


    def create_prt_content(self, prt_data, conditions, colors):
        header = '''FileVersion:        2

ResolutionOfTime:   Volumes

Experiment:         Associative learning

BackgroundColor:    0 0 0
TextColor:          255 255 255
TimeCourseColor:    255 255 255
TimeCourseThick:    3
ReferenceFuncColor: 0 0 80
ReferenceFuncThick: 3

'''

        content = f'NrOfConditions:     {len(conditions)}\n\n'
        for color_idx, condition in enumerate(prt_data.keys()):
            content += f'{condition}\n'
            content += f'{len(prt_data[condition])}\n'
            for trial in prt_data[condition]:
                content += f'\t{trial["start"]}   {trial["end"]}\n'
            content += f'Color: {colors[color_idx][0]} {colors[color_idx][1]} {colors[color_idx][2]}\n\n'


        return header + content

    def verify_TR_correctness(self, log_data, tr_list_filename, verbose=False):
        with open(tr_list_filename, 'rb') as f: tr_list = pkl.load(f)

        durations = np.array(log_data['Duration'])[np.where(np.array(log_data['Stim']) == 'PULSE')]
        pulses = np.array(log_data['Stim'])[np.where((np.array(log_data['Event']) == 'Action') | (np.array(log_data['Event']) == 'Sound'))]

        count_dups = [sum(1 for _ in group) for _, group in groupby(pulses)]
        count_dups = [elem for elem in count_dups if elem != 1]

        c = 0
        tr_counts = {'TR_list_after_sound':[], 'TR_list_after_choice':[]}
        pos = 'TR_list_after_sound'

        for c in count_dups:
            tr_counts[pos].append(c)
            if pos == 'TR_list_after_sound':
                pos = 'TR_list_after_choice'
            else:
                pos = 'TR_list_after_sound'

        if tr_counts == tr_list:
            if verbose: print('Experimental TR verfication was successfull!')
            return True
        print('Warning, logged TR\'s don\'t match with data')
        return False

    def verify_stim_correctness(sefl, log_data, stim_list_filename, verbose=False):
        with open(stim_list_filename, 'rb') as f: stim_list = pkl.load(f)

        pictures = np.array(log_data['Stim'])[np.where(np.array(log_data['Event']) == 'Picture')]
        exp = np.array(log_data['Exp'])[np.where(np.array(log_data['Event']) == 'Picture')]

        pics_with_exp = []
        for i,e in enumerate(exp):
            if e != '':
                s = pictures[i] + '=' + exp[i]
            else:
                s = pictures[i]
            pics_with_exp.append(s)

        pics_with_exp = np.array(pics_with_exp).reshape(len(pics_with_exp)//3, 3)

        sounds = np.array(log_data['Stim'])[np.where(np.array(log_data['Event']) == 'Sound')]

        stims = [list(np.append(p, sounds[i])) for i,p in enumerate(pics_with_exp)]

        if stims == stim_list:
            if verbose: print('Experimental stimulus verfication was successfull!')
            return True
        print('Warning, logged stimuli don\'t match with data')
        return False
