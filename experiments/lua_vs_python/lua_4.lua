local objects = find_objects()
local colour_counts = {}
for _, obj in ipairs(objects) do
    colour_counts[obj.colour] = (colour_counts[obj.colour] or 0) + 1
end
local rarest_colour = nil
local min_count = math.huge
for colour, count in pairs(colour_counts) do
    if count < min_count then
        min_count = count
        rarest_colour = colour
    end
end
local target = nil
for _, obj in ipairs(objects) do
    if obj.colour == rarest_colour then
        target = obj
        break
    end
end
if target then
    move_to(target.x, target.y)
end
