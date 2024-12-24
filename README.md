-- Script para teletransportar todos os jogadores para um novo mapa
-- Certifique-se de substituir 'SEU_PLACE_ID' pelo ID do lugar para onde você deseja teletransportar os jogadores

local TeleportService = game:GetService("TeleportService")
local Players = game:GetService("Players")
local TARGET_PLACE_ID = 130111483311364 -- Substitua pelo ID do seu lugar

-- Função para teletransportar todos os jogadores
local function teleportAllPlayers()
    local playersToTeleport = {}
    for _, player in pairs(Players:GetPlayers()) do
        table.insert(playersToTeleport, player)
    end
    TeleportService:TeleportPartyAsync(TARGET_PLACE_ID, playersToTeleport)
end

-- Chama a função para teletransportar todos os jogadores
teleportAllPlayers()
