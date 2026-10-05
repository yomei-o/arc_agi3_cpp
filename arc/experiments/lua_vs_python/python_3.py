for n in range(1, 6):
    press(n)
    objects = find_objects()
    say(len(objects))
