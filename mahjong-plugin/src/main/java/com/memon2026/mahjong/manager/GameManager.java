package com.memon2026.mahjong.manager;

import com.memon2026.mahjong.game.MahjongGame;
import com.memon2026.mahjong.game.GameState;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * 游戏房间管理器，负责创建/查找/销毁游戏房间。
 */
public class GameManager {

    /** 房间名 -> 游戏实例 */
    private final Map<String, MahjongGame> games = new HashMap<>();

    /** 玩家UUID -> 所在房间名（方便快速查找玩家在哪个房间） */
    private final Map<UUID, String> playerRoomMap = new HashMap<>();

    /**
     * 创建新房间。
     * @return null = 成功；否则为错误信息
     */
    public String createRoom(String roomName, UUID creatorUuid, String creatorName) {
        if (games.containsKey(roomName)) return "§c房间 §f" + roomName + " §c已存在";
        if (playerRoomMap.containsKey(creatorUuid)) {
            return "§c你已在房间 §f" + playerRoomMap.get(creatorUuid) + " §c中，请先离开";
        }
        MahjongGame game = new MahjongGame(roomName, creatorUuid);
        String err = game.addPlayer(creatorUuid, creatorName);
        if (err != null) return err;
        games.put(roomName, game);
        playerRoomMap.put(creatorUuid, roomName);
        return null;
    }

    /**
     * 玩家加入已有房间。
     * @return null = 成功；否则为错误信息
     */
    public String joinRoom(String roomName, UUID playerUuid, String playerName) {
        MahjongGame game = games.get(roomName);
        if (game == null) return "§c房间 §f" + roomName + " §c不存在";
        if (playerRoomMap.containsKey(playerUuid)) {
            String current = playerRoomMap.get(playerUuid);
            if (current.equals(roomName)) return "§c你已在该房间中";
            return "§c你已在房间 §f" + current + " §c中，请先用 §f/mj leave §c离开";
        }
        String err = game.addPlayer(playerUuid, playerName);
        if (err != null) return err;
        playerRoomMap.put(playerUuid, roomName);
        return null;
    }

    /**
     * 玩家离开房间。
     * @return null = 成功；否则为错误信息
     */
    public String leaveRoom(UUID playerUuid) {
        String roomName = playerRoomMap.get(playerUuid);
        if (roomName == null) return "§c你不在任何房间中";
        MahjongGame game = games.get(roomName);
        if (game != null) {
            game.removePlayer(playerUuid);
            if (game.getState() == GameState.WAITING && game.getPlayerCount() == 0) {
                games.remove(roomName);
            }
            // 如果游戏进行中有人离开，房间保留直到游戏结束
        }
        playerRoomMap.remove(playerUuid);
        return null;
    }

    /**
     * 获取玩家所在的游戏实例。
     */
    public MahjongGame getPlayerGame(UUID playerUuid) {
        String roomName = playerRoomMap.get(playerUuid);
        if (roomName == null) return null;
        return games.get(roomName);
    }

    /**
     * 获取玩家所在的房间名。
     */
    public String getPlayerRoom(UUID playerUuid) {
        return playerRoomMap.get(playerUuid);
    }

    /**
     * 根据房间名获取游戏实例。
     */
    public MahjongGame getGame(String roomName) {
        return games.get(roomName);
    }

    /**
     * 获取所有房间。
     */
    public Collection<MahjongGame> getAllGames() {
        return games.values();
    }

    /**
     * 清理已结束的空房间。
     */
    public void cleanupEndedRooms() {
        List<String> toRemove = new ArrayList<>();
        for (Map.Entry<String, MahjongGame> entry : games.entrySet()) {
            MahjongGame g = entry.getValue();
            if (g.getState() == GameState.ENDED && g.getPlayerCount() == 0) {
                toRemove.add(entry.getKey());
            }
        }
        toRemove.forEach(games::remove);
    }
}
