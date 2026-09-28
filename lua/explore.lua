-- A first policy, written the way a model would be asked to write one.
--
-- It does not know the rules. It finds the thing that moves when a button is
-- pressed, learns which buttons move it, and then walks that thing onto
-- whatever object it has not visited yet. That is the cheapest hypothesis that
-- covers most of these games, and it is the one the 27B stated unprompted for
-- ka59: "the player controls the F object, navigate it to the + cells".

local me_colour, me_area = nil, nil
local moves = {}                 -- button -> {dx, dy}

local function find_me()
  if not me_colour then return nil end
  for _, o in ipairs(objects()) do
    if o.colour == me_colour and o.area == me_area then return o end
  end
end

-- Press each button once and watch what moves. Whatever moved is us.
local buttons = {}
for _, n in ipairs(actions()) do
  if n >= 1 and n <= 5 then buttons[#buttons + 1] = n end
end

for _, n in ipairs(buttons) do
  press(n)
  for _, e in ipairs(changes()) do
    if e.kind == "moved" then
      me_colour, me_area = e.colour, e.area
      moves[n] = {dx = e.dx, dy = e.dy}
      break
    end
  end
end

if not me_colour then
  say("nothing moves under the buttons; this is a clicking game")
  for _, o in ipairs(objects()) do click(o.x, o.y) end
  return
end

say("I am the colour-" .. me_colour .. " object of " .. me_area .. " cells")
local n = 0
for b, v in pairs(moves) do
  say("  button " .. b .. " moves me (" .. v.dx .. "," .. v.dy .. ")")
  n = n + 1
end

-- Walk to each other object in turn, greedily, giving up on one that will not
-- come closer after a few tries.
local visited = {}
for round = 1, 200 do
  local me = find_me()
  if not me then break end
  local target
  for _, o in ipairs(objects()) do
    local key = o.colour .. ":" .. o.x .. ":" .. o.y
    if not visited[key] and not (o.colour == me_colour and o.area == me_area) then
      target = o
      visited[key] = true
      break
    end
  end
  if not target then break end

  local stuck = 0
  for try = 1, 60 do
    me = find_me()
    if not me then break end
    if me.x == target.x and me.y == target.y then break end
    local best, bestd = nil, math.huge
    for b, v in pairs(moves) do
      local d = math.abs(me.x + v.dx - target.x) + math.abs(me.y + v.dy - target.y)
      if d < bestd then best, bestd = b, d end
    end
    if not best then break end
    local before = level()
    if not press(best) then
      stuck = stuck + 1
      if stuck > 3 then break end
    else
      stuck = 0
    end
    if level() ~= before then say("level changed at step " .. steps()) end
  end
end
say("finished at level " .. level() .. " after " .. steps() .. " actions")
