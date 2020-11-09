function part_names = getlabel(cate)
    type = cate;
    if strcmp(type, 'chair') == 1
        part_names = {'back', 'seat', 'hand_1', 'hand_2', 'leg_ver_1', 'leg_ver_2', 'leg_ver_3', 'leg_ver_4'};
    elseif strcmp(type, 'knife') == 1
        part_names = {'part1', 'part2'};
    elseif strcmp(type, 'guitar') == 1
        part_names = {'part1', 'part2', 'part3'};
    elseif strcmp(type, 'monitor') == 1
        part_names = {'display', 'connector', 'base'};
    elseif strcmp(type, 'skateboard')
        part_names = {'surface', 'bearing1', 'bearing2', 'wheel1_1', 'wheel1_2', 'wheel2_1', 'wheel2_2'};
    elseif strcmp(type, 'cup') == 1
        part_names = {'part1', 'part2'};
    elseif strcmp(type, 'car') == 1
        part_names = {'body', 'left_front_wheel', 'right_front_wheel', 'left_back_wheel', 'right_back_wheel', 'left_mirror', 'right_mirror'};
    elseif strcmp(type, 'plane') == 1
        part_names = {'body', 'left_wing', 'right_wing', 'left_tail', 'right_tail', 'up_tail', 'down_tail', 'front_gear', 'left_gear', 'right_gear', 'left_engine1', 'right_engine1', 'left_engine2', 'right_engine2'};
    elseif strcmp(type, 'table') == 1
        part_names = {'surface', 'left_leg1', 'left_leg2', 'left_leg3', 'left_leg4', 'right_leg1', 'right_leg2', 'right_leg3', 'right_leg4'};
    end
end