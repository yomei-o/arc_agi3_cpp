-- Learn the buttons, then let the host do the walking.
for _, n in ipairs(actions()) do
  if n >= 1 and n <= 5 then press(n) end
end

local m = me()
if not m then
  say("nothing moved under the buttons")
  return
end
say("I am the colour-" .. m.colour .. " object of " .. m.area .. " cells at " .. m.x .. "," .. m.y)

-- Visit the other objects, nearest first. move_to spends the actions.
local targets = {}
for _, o in ipairs(objects()) do
  if not (o.colour == m.colour and o.area == m.area) then targets[#targets + 1] = o end
end
table.sort(targets, function(a, b)
  return math.abs(a.x - m.x) + math.abs(a.y - m.y) < math.abs(b.x - m.x) + math.abs(b.y - m.y)
end)

for i, o in ipairs(targets) do
  if steps() > 600 then break end
  local before = level()
  local ok = move_to(o.x, o.y, 80)
  say((ok and "reached " or "gave up on ") .. o.glyph .. " at " .. o.x .. "," .. o.y
      .. "  steps=" .. steps() .. "  walls=" .. #walls())
  if level() > before then
    say("LEVEL UP at " .. steps() .. " actions")
    return
  end
end
say("done at level " .. level() .. " after " .. steps() .. " actions")
