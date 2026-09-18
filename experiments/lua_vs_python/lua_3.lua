for i = 1, 5 do
    press(i)
    local objs = find_objects()
    say(#objs)
end
