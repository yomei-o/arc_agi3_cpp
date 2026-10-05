local my_x, my_y
local objects = find_objects()
for _, obj in ipairs(objects) do
    if board().get(obj.x, obj.y) == obj.colour then
        my_x, my_y = obj.x, obj.y
        break
    end
end
for i = 1, 200 do
    local nearest = nil
    local min_dist = math.huge
    for _, obj in ipairs(find_objects()) do
        if obj.x == my_x and obj.y == my_y then
            goto continue
        end
        local dist = math.abs(obj.x - my_x) + math.abs(obj.y - my_y)
        if dist < min_dist then
            min_dist = dist
            nearest = obj
        end
        ::continue::
    end
    if nearest then
        move_to(nearest.x, nearest.y)
        my_x, my_y = nearest.x, nearest.y
    else
        break
    end
end
