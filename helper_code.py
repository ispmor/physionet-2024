#!/usr/bin/env python

# Do *not* edit this script.
# These are helper functions that you can use with your code.
# Check the example code to see how to import these functions to your code.

from re import A, I
from typing import Tuple
import numpy as np
import os
import sys
import cv2
from pandas import DataFrame
from pandas._libs.lib import is_datetime_with_singletz_array
from requests.models import LocationParseError
import pytesseract
### Challenge data I/O functions


leads = ['i', 'ii', 'iii', 'avr', 'avf', 'avl', 'v1', 'v2', 'v3', 'v4', 'v5',
         'v6']

ratio=0.75 #4/3
target_height = 128
target_width = int(target_height * ratio)
# Find the records in a folder and its subfolders.
def find_records(folder):
    records = set()
    for root, directories, files in os.walk(folder):
        for file in files:
            extension = os.path.splitext(file)[1]
            if extension == '.hea':
                record = os.path.relpath(os.path.join(root, file), folder)[:-4]
                records.add(record)
    records = sorted(records)
    return records

# Load the header for a record.
def load_header(record):
    header_file = get_header_file(record)
    header = load_text(header_file)
    return header

# Load the signal(s) for a record.
def load_signal(record):
    import wfdb

    signal_files = get_signal_files(record)
    if signal_files:
        signal, fields = wfdb.rdsamp(record)
    else:
        signal, fields = None, None
    return signal, fields

def load_signals(record):
    return load_signal(record)


def index_of_row_in_lines(row_top, lines, distance_avg):
    for idx, l in enumerate(lines):
        if abs(row_top - ((l[1] + l[3]) // 2)) < (distance_avg // 2):
            return idx

    return -1

# Load the image(s) for a record.
def load_image(record):
    path = os.path.split(record)[0]
    image_files = get_image_files(record)

    images = list()
    for image_file in image_files:
        image_file_path = os.path.join(path, image_file)
        if os.path.isfile(image_file_path):
            #Load image
            image = cv2.imread(image_file_path)#cv2.IMREAD_GRAYSCALE)
            #image = cv2.resize(image, (0,0), fx = 0.5, fy = 0.5)
            image = clean_up_image(image)
            image, lines = adjust_rotation(image)
            max_h, max_w = image.shape
            importantLines = list(filter(lambda line:
                                         abs(getAngleFromPoints(line)) - 180 <= 2.5,
                                         lines))
            onlySingularLines =  getLinesNotCoveringEachOther(importantLines, 32)

            distances = []
            if len(onlySingularLines) < 2:
                distances = [max_h * 0.5]
            else:
                for i in range(1, len(onlySingularLines)):
                    distances.append(((onlySingularLines[i][1] + onlySingularLines[i][3])
                            // 2 -
                            (onlySingularLines[i-1][1]+onlySingularLines[i-1][3])
                             // 2))
            # TODO naive approach to distanes assuming they are equal
            distance_avg = np.average(distances)


            image =cv2.medianBlur(image, 3)

            image_for_ocr = prepare_for_ocr(image,
                                            onlySingularLines,64)


            ocr_data: DataFrame = pytesseract.image_to_data(
                    image_for_ocr, 'eng',
                    config="-c tessedit_char_whitelist=123456IaAvVfFrRlL --psm 12",
                    output_type="data.frame")


            ocr_data.dropna(0, inplace=True, subset=['text'])
            dict_flattened_images = {}
            if ocr_data.empty:
                for lead_name in leads:
                    dict_flattened_images[lead_name]=np.zeros((target_height * target_width,))
            else:
                dict_flattened_images = build_input(image, ocr_data, onlySingularLines, distance_avg, max_w, max_h)

            print(dict_flattened_images)
            images.append(image)

    return images


def build_input(image, ocr_data, onlySingularLines, distance_avg, max_w, max_h):
    sorted_dict_leads =extract_sorted_ocr_results(ocr_data, onlySingularLines, distance_avg)
    dict_of_cropped_images = extract_cropped_images_for_leads(image, sorted_dict_leads, onlySingularLines, distance_avg, max_w, max_h)
    dict_flattened_images = {}

    for lead_name in leads:
        if lead_name in dict_of_cropped_images:
            dict_flattened_images[lead_name] = keepOnlyBiggestObject(dict_of_cropped_images[lead_name]).flatten()
        else:
            dict_flattened_images[lead_name]=np.zeros((target_height * target_width,))
    return dict_flattened_images



def extract_cropped_images_for_leads(image, sorted_dict_leads, onlySingularLines, distance_avg, max_w, max_h):
    dict_of_cropped_images = {}
    for i in range(0, len(sorted_dict_leads)-1):
        idx, row = sorted_dict_leads[i][1]
        _, row_next = sorted_dict_leads[i+1][1]
        lead_name = sorted_dict_leads[i][0]
        line_idx = int(row['line'])
        line_exists = line_idx >= 0
        line = onlySingularLines[int(row['line'])] if line_idx >= 0 else  None
        line_y = 0
        local_min_h = 0
        local_max_h = max_h
        local_min_w = 0
        local_max_w = max_w

        if line_exists:
            line_y =int((line[1] + line[3])//2)
        else:
            line_y = row['top']

        if line_y + distance_avg * 0.7 > max_h:
            local_max_h:int = max_h
        else:
            local_max_h:int =  int(line_y + distance_avg * 0.7)

        if line_y - distance_avg*0.7 < 0:
            local_min_h = 0
        else:
            local_min_h =  int(line_y - distance_avg*0.7)

        #same line scenario
        if abs(row['top'] - row_next['top']) < 15:
            local_min_w = row['left']
            local_max_w = row_next['left']
        else:
            #next lead in new line scenario:
            local_min_w = row['left']
            local_max_w = max_w

        dict_of_cropped_images[lead_name] = image[local_min_h:local_max_h, local_min_w: local_max_w] 
    return dict_of_cropped_images



def extract_sorted_ocr_results(ocr_data, onlySingularLines, distance_avg):
    ocr_data['text']=ocr_data['text'].apply(lambda x: str(x).lower())
    ocr_data['text_as_set'] = ocr_data['text'].apply(lambda x: set(x))
    ocr_data['line'] = ocr_data['top'].apply(lambda x: index_of_row_in_lines(x, onlySingularLines, distance_avg))
    ocr_data.sort_values(['line', 'left'], inplace=True, ignore_index=True)
    dict_leads = {}
    for idx, row in ocr_data.iterrows():
        for lead in leads:
            if row['text'] == lead or (
                    set(lead).issubset(row['text_as_set']) and
                    row['text'].endswith('i') and
                    len(row['text_as_set']) >2 and len(set(lead)) > 1) :
                dict_leads[lead] = (idx,row)
    return sorted(dict_leads.items(), key=lambda x: x[1][0])


def keepOnlyBiggestObject(image):
    # Generate intermediate image; use morphological closing to keep parts of the brain together
    inter = cv2.morphologyEx(image, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Find largest contour in intermediate image
    cnts, _ = cv2.findContours(inter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)

    out = np.zeros(image.shape, np.uint8)
    cv2.drawContours(out, [cnt], -1, 255, cv2.FILLED)

    out = cv2.bitwise_and(image, out)
    out = image_dilation(out)
    out= cv2.resize(out, (target_height, target_width))
    return out


def prepare_for_ocr(image, baselines, h):
    res = image.copy()
    w = image.shape[1]
    neg_h = image.shape[0]//7
    for line in baselines:
        y = (line[1] + line [3]) // 2
        res = cv2.rectangle(res, (0, y-neg_h), (w-1, (y+h//2)), (0, 0, 0), -1)
    return res





def getLinesNotCoveringEachOther(lines, maxgap):
    lines_avg_y = list(map(lambda line: (line, (line[0][1] + line[0][3]) / 2), lines))
    lines_avg_y.sort(key=lambda entry: entry[1])
    groups = [[lines_avg_y[0]]]
    for line_tuple in lines_avg_y[1:]:
        if abs(line_tuple[1] - groups[-1][-1][1]) <= maxgap:
            groups[-1].append(line_tuple)
        else:
            groups.append([line_tuple])

    avg_groups = []
    for g in groups:
        only_arrays = np.concatenate([x[0] for x in g])
        avg_groups.append(np.average(only_arrays, axis=0).astype(int))
    return avg_groups

def getAngleFromPoints(line: Tuple) -> float:
    x1, y1, x2, y2 = line[0]
    angle: float = np.arctan2(y1 - y2, x1 - x2) * 180 / np.pi
    return angle


def adjust_rotation(image):
    canimg = cv2.Canny(image, 50, 150,apertureSize = 3)
    lines= cv2.HoughLines(canimg, 1, np.pi/180.0, 250, np.array([]))
    rho, theta = lines[0][0]
    image = rotate_image(image, 180*theta/3.1415926 - 90)
    
    min_length = int(image.shape[1] * .75)
    max_gap = int(image.shape[1] * .15)
    
    canimg = cv2.Canny(image, 50, 150,apertureSize = 3)
    lines = cv2.HoughLinesP(\
            canimg,1,np.pi/180,min_length//7,minLineLength=min_length,maxLineGap=max_gap)
    return (image, lines)



def debug_lines(image):
    img = image.copy()
    min_length = int(img.shape[1] * .75)
    max_gap = int(img.shape[1] * .15)
    canimg = cv2.Canny(image, 50, 150,apertureSize = 3)
    lines = cv2.HoughLinesP(\
            canimg,1,np.pi/180,min_length//7,minLineLength=min_length,maxLineGap=max_gap)
    print(lines)

    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.imshow('LINES', img)
    cv2.waitKey(0)

def clean_up_image(image):
    image[:,:,1]=0
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image_erosion(image)
    image = image_dilation(image)
    image = image_binarisation(image)
    return image

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1],
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    return result


def display(image):
    cv2.imshow('img', image)
    cv2.waitKey(0)


def image_dilation(image, iterations=1):
    kernel = np.ones((3,3), np.uint8)
    return cv2.dilate(image, kernel, iterations=iterations)


def image_binarisation(image):
    ret3,th3 = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return th3



def image_erosion(image, iterations=1):
    kernel = np.ones((3, 3), np.uint8)
    # Using cv2.erode() method 
    return cv2.erode(image, kernel, iterations=iterations)

def load_images(record):
    return load_image(record)

# Load the dx class(es) for a record.
def load_dx(record):
    header = load_header(record)
    dx = get_dxs_from_header(header)
    return dx

def load_dxs(record):
    return load_dx(record)

# Save the header for a record.
def save_header(record, header):
    header_file = get_header_file(record)
    save_text(header_file, header)

# Save the signal(s) for a record.
def save_signal(record, signal, comments=list()):
    header = load_header(record)
    path, record = os.path.split(record)
    sampling_frequency = get_sampling_frequency(header)
    signal_formats = get_signal_formats(header)
    adc_gains = get_adc_gains(header)
    baselines = get_baselines(header)
    signal_units = get_signal_units(header)
    signal_names = get_signal_names(header)

    if all(signal_format == '16' for signal_format in signal_formats):
        signal = np.clip(signal, -2**15 + 1, 2**15 - 1)
        signal = np.asarray(signal, dtype=np.int16)
    else:
        signal_format_string = ', '.join(sorted(set(signal_formats)))
        raise NotImplementedError(f'{signal_format_string} not implemented')

    import wfdb
    wfdb.wrsamp(record, fs=sampling_frequency, units=signal_units, sig_name=signal_names, \
                d_signal=signal, fmt=signal_formats, adc_gain=adc_gains, baseline=baselines, comments=comments, \
                write_dir=path)

def save_signals(record, signals):
    save_signal(record, signals)

# Save the dx class(es) for a record.
def save_dx(record, dx):
    header_file = get_header_file(record)
    header = load_text(header_file)
    header += '#Dx: ' + ', '.join(dx) + '\n'
    save_text(header_file, header)
    return header

def save_dxs(record, dxs):
    return save_dx(record, dxs)

### Helper Challenge functions

# Load a text file as a string.
def load_text(filename):
    with open(filename, 'r') as f:
        string = f.read()
    return string

# Save a string as a text file.
def save_text(filename, string):
    with open(filename, 'w') as f:
        f.write(string)

# Get the record name from a header file.
def get_record_name(string):
    value = string.split('\n')[0].split(' ')[0].split('/')[0].strip()
    return value

# Get the number of signals from a header file.
def get_num_signals(string):
    value = string.split('\n')[0].split(' ')[1].strip()
    if is_integer(value):
        value = int(value)
    else:
        value = None
    return value

# Get the sampling frequency from a header file.
def get_sampling_frequency(string):
    value = string.split('\n')[0].split(' ')[2].split('/')[0].strip()
    if is_number(value):
        value = float(value)
    else:
        value = None
    return value

# Get the number of samples from a header file.
def get_num_samples(string):
    value = string.split('\n')[0].split(' ')[3].strip()
    if is_integer(value):
        value = int(value)
    else:
        value = None
    return value

# Get signal units from a header file.
def get_signal_formats(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[1]
            if 'x' in field:
                field = field.split('x')[0]
            if ':' in field:
                field = field.split(':')[0]
            if '+' in field:
                field = field.split('+')[0]
            value = field
            values.append(value)
    return values

# Get signal units from a header file.
def get_adc_gains(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[2]
            if '/' in field:
                field = field.split('/')[0]
            if '(' in field and ')' in field:
                field = field.split('(')[0]
            value = float(field)
            values.append(value)
    return values

# Get signal units from a header file.
def get_baselines(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[2]
            if '/' in field:
                field = field.split('/')[0]
            if '(' in field and ')' in field:
                field = field.split('(')[1].split(')')[0]
            value = int(field)
            values.append(value)
    return values

# Get signal units from a header file.
def get_signal_units(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            field = l.split(' ')[2]
            if '/' in field:
                value = field.split('/')[1]
            else:
                value = 'mV'
            values.append(value)
    return values

# Get the number of samples from a header file.
def get_signal_names(string):
    num_signals = get_num_signals(string)
    values = list()
    for i, l in enumerate(string.split('\n')):
        if 1 <= i <= num_signals:
            value = l.split(' ')[8]
            values.append(value)
    return values

# Get a variable from a string.
def get_variable(string, variable_name):
    variable = ''
    has_variable = False
    for l in string.split('\n'):
        if l.startswith(variable_name):
            variable = l[len(variable_name):].strip()
            has_variable = True
    return variable, has_variable

# Get variables from a text file.
def get_variables(string, variable_name, sep=','):
    variables = list()
    has_variable = False
    for l in string.split('\n'):
        if l.startswith(variable_name):
            variables += [variable.strip() for variable in l[len(variable_name):].strip().split(sep)]
            has_variable = True
    return variables, has_variable

# Get the signal file(s) from a header or a similar string.
def get_signal_files_from_header(string):
    signal_files = list()
    for i, l in enumerate(string.split('\n')):
        arrs = [arr.strip() for arr in l.split(' ')]
        if i==0 and not l.startswith('#'):
            num_channels = int(arrs[1])
        elif i<=num_channels and not l.startswith('#'):
            signal_file = arrs[0]
            if signal_file not in signal_files:
                signal_files.append(signal_file)
        else:
            break
    return signal_files

# Get the image file(s) from a header or a similar string.
def get_image_files_from_header(string):
    images, has_image = get_variables(string, '#Image:')
    if not has_image:
        raise Exception('No images available: did you forget to generate or include the images?')
    return images

# Get the dx class(es) from a header or a similar string.
def get_dxs_from_header(string):
    dxs, has_dx = get_variables(string, '#Dx:')
    if not has_dx:
        raise Exception('No dx classes available: are you trying to load the classes from the held-out dataset, or did you forget to prepare the data to include the classes?')
    return dxs

# Get the header file for a record.
def get_header_file(record):
    if not record.endswith('.hea'):
        header_file = record + '.hea'
    else:
        header_file = record
    return header_file

# Get the signal file(s) for a record.
def get_signal_files(record):
    header_file = get_header_file(record)
    header = load_text(header_file)
    signal_files = get_signal_files_from_header(header)
    return signal_files

# Get the image file(s) for a record.
def get_image_files(record):
    header_file = get_header_file(record)
    header = load_text(header_file)
    image_files = get_image_files_from_header(header)
    return image_files

### Evaluation functions

# Construct the binary one-vs-rest confusion matrices, where the columns are the expert labels and the rows are the classifier
# for the given classes.
def compute_one_vs_rest_confusion_matrix(labels, outputs, classes):
    assert np.shape(labels) == np.shape(outputs)

    num_instances = len(labels)
    num_classes = len(classes)

    A = np.zeros((num_classes, 2, 2))
    for i in range(num_instances):
        for j in range(num_classes):
            if labels[i, j] == 1 and outputs[i, j] == 1: # TP
                A[j, 0, 0] += 1
            elif labels[i, j] == 0 and outputs[i, j] == 1: # FP
                A[j, 0, 1] += 1
            elif labels[i, j] == 1 and outputs[i, j] == 0: # FN
                A[j, 1, 0] += 1
            elif labels[i, j] == 0 and outputs[i, j] == 0: # TN
                A[j, 1, 1] += 1

    return A

# Compute macro F-measure.
def compute_f_measure(labels, outputs):
    # Compute confusion matrix.
    classes = sorted(set.union(*map(set, labels)))
    labels = compute_one_hot_encoding(labels, classes)
    outputs = compute_one_hot_encoding(outputs, classes)
    A = compute_one_vs_rest_confusion_matrix(labels, outputs, classes)

    num_classes = len(classes)
    per_class_f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 0, 0], A[k, 0, 1], A[k, 1, 0], A[k, 1, 1]
        if 2 * tp + fp + fn > 0:
            per_class_f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            per_class_f_measure[k] = float('nan')

    if np.any(np.isfinite(per_class_f_measure)):
        macro_f_measure = np.nanmean(per_class_f_measure)
    else:
        macro_f_measure = float('nan')

    return macro_f_measure, per_class_f_measure, classes

# Reorder channels in signal.
def reorder_signal(input_signal, input_channels, output_channels):
    if input_signal is None:
        return None

    if input_channels == output_channels and len(set(input_channels)) == len(set(output_channels)) == len(output_channels):
        output_signal = input_signal
    else:
        input_channels = [channel.strip().casefold() for channel in input_channels]
        output_channels = [channel.strip().casefold() for channel in output_channels]

        num_samples = np.shape(input_signal)[0]
        num_channels = len(output_channels)
        data_type = input_signal.dtype
        output_signal = np.zeros((num_samples, num_channels), dtype=data_type)

        for i, output_channel in enumerate(output_channels):
            for j, input_channel in enumerate(input_channels):
                if input_channel == output_channel:
                    output_signal[:, i] += input_signal[:, j]

    return output_signal

# Pad or truncate signal.
def trim_signal(input_signal, num_samples):
    if input_signal is None:
        return None

    cur_samples, num_channels = np.shape(input_signal)
    data_type = input_signal.dtype

    if cur_samples == num_samples:
        output_signal = input_signal
    else:
        output_signal = np.zeros((num_samples, num_channels), dtype=data_type)
        if cur_samples < num_samples: # Zero-pad the signals.
            output_signal[:cur_samples, :] = input_signal
        else: # Truncate the signals.
            output_signal = input_signal[:num_samples, :]

    return output_signal

# Compute SNR.
def compute_snr(label_signal, output_signal):
    if label_signal is None or output_signal is None:
        return None

    assert(np.all(np.shape(label_signal) == np.shape(output_signal)))

    label_signal[np.isnan(label_signal)] = 0
    output_signal[np.isnan(output_signal)] = 0

    noise_signal = output_signal - label_signal

    x = np.sum(label_signal**2)
    y = np.sum(noise_signal**2)

    if y > 0:
        snr = 10 * np.log10(x / y)
    else:
        snr = float('inf')

    return snr

### Other helper functions

# Check if a variable is a number or represents a number.
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

# Check if a variable is an integer or represents an integer.
def is_integer(x):
    if is_number(x):
        return float(x).is_integer()
    else:
        return False

# Check if a variable is a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False

# Check if a variable is a NaN (not a number) or represents a NaN.
def is_nan(x):
    if is_number(x):
        return np.isnan(float(x))
    else:
        return False

# Cast a value to an integer if an integer, a float if a non-integer float, and an unknown value otherwise.
def cast_int_float_unknown(x):
    if is_integer(x):
        x = int(x)
    elif is_finite_number(x):
        x = float(x)
    elif is_number(x):
        x = 'Unknown'
    else:
        raise NotImplementedError(f'Unable to cast {x}.')
    return x

# Construct the one-hot encoding of data for the given classes.
def compute_one_hot_encoding(data, classes):
    num_instances = len(data)
    num_classes = len(classes)

    one_hot_encoding = np.zeros((num_instances, num_classes), dtype=np.bool_)
    unencoded_data = list()
    for i, x in enumerate(data):
        for y in x:
            for j, z in enumerate(classes):
                if (y == z) or (is_nan(y) and is_nan(z)):
                    one_hot_encoding[i, j] = 1

    return one_hot_encoding
