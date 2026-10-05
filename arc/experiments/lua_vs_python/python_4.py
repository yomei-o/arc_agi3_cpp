objects = find_objects()
if not objects:
    pass
else:
    # Count colours among objects
    colour_counts = {}
    for obj in objects:
        c = obj.colour
        colour_counts[c] = colour_counts.get(c, 0) + 1
    # Find the rarest colour (minimum count)
    rarest_colour = None
    min_count = float('inf')
    for c, count in colour_counts.items():
        if count < min_count:
            min_count = count
            rarest_colour = c
    # Find the first object with that rarest colour
    target = None
    for obj in objects:
        if obj.colour == rarest_colour:
            target = obj
            break
    if target:
        move_to(target.x, target.y)
