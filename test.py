num_good_time = 0
hours = 0
minutes = 0
input = '987654321'  # input minutes
total_minutes = 0
while True:
    if eval(input) >= total_minutes:
        # print("time: ", hours, ':', '{:02d}'.format(minutes))
        time_str = str(hours) + '{:02d}'.format(minutes)
        if len(time_str) == 3:
            if eval(time_str[-1]) - eval(time_str[-2]) == eval(time_str[-2]) - eval(time_str[-3]):
                num_good_time += 1
                # print(str(hours) + '{:02d}'.format(minutes))
        if len(time_str) == 4:
            if eval(time_str[-1]) - eval(time_str[-2]) == eval(time_str[-2]) - eval(time_str[-3]) == eval(time_str[-3]) - eval(time_str[-4]):
                num_good_time += 1
                # print(str(hours) + '{:02d}'.format(minutes))

        minutes += 1
        total_minutes += 1
        if minutes == 60:
            hours += 1
            minutes = 0
        if hours == 24:
            hours = 0

    else:
        print(num_good_time)
        break
