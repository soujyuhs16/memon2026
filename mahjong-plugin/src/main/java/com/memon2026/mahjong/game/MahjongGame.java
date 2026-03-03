package com.memon2026.mahjong.game;

import org.bukkit.Bukkit;
import org.bukkit.entity.Player;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * 广东麻将游戏核心逻辑
 *
 * 规则说明：
 * - 108张牌：万/条/饼各1-9点，每种4副
 * - 最多4人，最少2人（测试用）
 * - 广东麻将不允许吃牌（不能吃他人弃牌组顺子）
 * - 胡牌需满足缺一门
 * - 支持碰、明杠、暗杠、补杠、自摸、放炮
 */
public class MahjongGame {

    private final String roomName;
    private final UUID creatorUuid;
    private final List<GamePlayer> players;
    private GameState state;
    private final LinkedList<Tile> wall;

    private int currentPlayerIndex;
    private Tile lastDiscard;
    private int lastDiscardPlayerIndex;

    /** 当前等待认领（碰/杠/胡）的玩家集合：UUID -> 是否已决定 */
    private final Map<UUID, Boolean> claimDecisions;
    private boolean waitingForClaims;

    private GamePlayer winner;
    private boolean lastWinSelfDraw;

    public MahjongGame(String roomName, UUID creatorUuid) {
        this.roomName = roomName;
        this.creatorUuid = creatorUuid;
        this.players = new ArrayList<>();
        this.state = GameState.WAITING;
        this.wall = new LinkedList<>();
        this.claimDecisions = new HashMap<>();
    }

    // ---- Accessors ----

    public String getRoomName() { return roomName; }
    public UUID getCreatorUuid() { return creatorUuid; }
    public List<GamePlayer> getPlayers() { return players; }
    public GameState getState() { return state; }
    public int getCurrentPlayerIndex() { return currentPlayerIndex; }
    public GamePlayer getCurrentPlayer() { return players.get(currentPlayerIndex); }
    public Tile getLastDiscard() { return lastDiscard; }
    public boolean isWaitingForClaims() { return waitingForClaims; }
    public GamePlayer getWinner() { return winner; }
    public boolean isLastWinSelfDraw() { return lastWinSelfDraw; }
    public int getWallSize() { return wall.size(); }

    // ---- Room management ----

    /**
     * 玩家加入房间。
     * @return null = 成功；否则为错误信息
     */
    public String addPlayer(UUID uuid, String name) {
        if (state != GameState.WAITING) return "§c游戏已在进行中，无法加入";
        if (players.size() >= 4) return "§c房间已满（最多4人）";
        for (GamePlayer p : players) {
            if (p.getUuid().equals(uuid)) return "§c你已在该房间中";
        }
        GamePlayer gp = new GamePlayer(uuid, name);
        gp.setSeatWind(players.size()); // 按加入顺序分配座位风
        players.add(gp);
        return null;
    }

    public void removePlayer(UUID uuid) {
        players.removeIf(p -> p.getUuid().equals(uuid));
    }

    public GamePlayer getGamePlayer(UUID uuid) {
        for (GamePlayer p : players) {
            if (p.getUuid().equals(uuid)) return p;
        }
        return null;
    }

    public boolean isFull() { return players.size() >= 4; }
    public int getPlayerCount() { return players.size(); }

    // ---- Game start ----

    /**
     * 开始游戏（需要2-4名玩家）。
     * @return null = 成功；否则为错误信息
     */
    public String startGame() {
        if (state != GameState.WAITING) return "§c游戏已开始";
        if (players.size() < 2) return "§c至少需要2名玩家才能开始";

        state = GameState.PLAYING;
        buildWall();

        // 每人发13张，东家额外1张
        for (int round = 0; round < 13; round++) {
            for (GamePlayer p : players) {
                p.getHand().add(wall.poll());
            }
        }
        players.get(0).getHand().add(wall.poll()); // 东家多1张

        currentPlayerIndex = 0;
        waitingForClaims = false;
        lastDiscard = null;

        // 通知所有玩家
        broadcastMessage("§6§l======= 广东麻将开始！=======");
        broadcastMessage("§e房间: §f" + roomName + "  §e玩家: §f" + players.size() + "人  §e剩余牌墙: §f" + wall.size() + "张");
        for (GamePlayer p : players) {
            Player bp = Bukkit.getPlayer(p.getUuid());
            if (bp != null) {
                bp.sendMessage("§6你的座位: §e" + p.getSeatWindName() + "家");
                sendHandToPlayer(p);
            }
        }
        broadcastMessage("§e【东家】" + getCurrentPlayer().getName() + " §f请出牌  §7(/mj discard <牌序号或牌名>)");
        return null;
    }

    private void buildWall() {
        wall.clear();
        for (TileSuit suit : TileSuit.values()) {
            for (int num = 1; num <= 9; num++) {
                for (int i = 0; i < 4; i++) {
                    wall.add(new Tile(suit, num));
                }
            }
        }
        Collections.shuffle(wall);
    }

    // ---- Player actions ----

    /**
     * 当前玩家出牌。
     * @param playerUuid 玩家UUID
     * @param tile       要出的牌
     * @return null = 成功；否则为错误信息
     */
    public String discard(UUID playerUuid, Tile tile) {
        if (state != GameState.PLAYING) return "§c游戏未在进行中";
        if (waitingForClaims) return "§c请等待其他玩家决定碰/杠/胡";
        GamePlayer player = getGamePlayer(playerUuid);
        if (player == null) return "§c你不在该游戏中";
        if (players.indexOf(player) != currentPlayerIndex) return "§c还没轮到你出牌";
        if (!player.removeTile(tile)) return "§c你没有这张牌: §f" + tile;

        lastDiscard = tile;
        lastDiscardPlayerIndex = currentPlayerIndex;

        broadcastMessage("§e" + player.getName() + " §f打出: " + tile.getDisplay()
                + "  §7(牌墙剩余: " + wall.size() + "张)");

        // 检查其他玩家是否可碰/杠/胡
        boolean anyCanClaim = checkClaimsAfterDiscard(tile);
        if (anyCanClaim) {
            waitingForClaims = true;
            broadcastMessage("§7其他玩家可进行操作，请在 §e/mj peng §7/§e /mj gang §7/§e /mj hu §7/§e /mj pass §7中选择");
        } else {
            advanceToNextPlayer();
        }
        return null;
    }

    /** 检查并通知可以碰/杠/胡的玩家 */
    private boolean checkClaimsAfterDiscard(Tile tile) {
        claimDecisions.clear();
        boolean anyCanClaim = false;

        for (int i = 0; i < players.size(); i++) {
            if (i == lastDiscardPlayerIndex) continue;
            GamePlayer p = players.get(i);
            boolean canClaim = false;
            StringBuilder hints = new StringBuilder();

            // 能碰？（手中有2张相同）
            if (p.canPeng(tile)) {
                canClaim = true;
                hints.append("§a碰(§f/mj peng§a) ");
            }
            // 能杠？（手中有3张相同）
            if (p.canKong(tile)) {
                canClaim = true;
                hints.append("§6明杠(§f/mj gang§6) ");
            }
            // 能胡？
            List<Tile> testHand = new ArrayList<>(p.getHand());
            testHand.add(tile);
            if (WinChecker.canWin(testHand, p.getMelds())) {
                canClaim = true;
                hints.append("§c§l胡(§f/mj hu§c§l) ");
            }

            if (canClaim) {
                claimDecisions.put(p.getUuid(), false);
                anyCanClaim = true;
                Player bp = Bukkit.getPlayer(p.getUuid());
                if (bp != null) {
                    bp.sendMessage("§e======= 操作提示 =======");
                    bp.sendMessage("§f" + players.get(lastDiscardPlayerIndex).getName()
                            + " 打出: " + tile.getDisplay());
                    bp.sendMessage("§e你可以: " + hints);
                    bp.sendMessage("§7或放弃: §f/mj pass");
                }
            }
        }
        return anyCanClaim;
    }

    /**
     * 玩家碰牌（将他人弃牌 + 手中2张组成刻子）。
     * @return null = 成功；否则为错误信息
     */
    public String peng(UUID playerUuid) {
        if (!waitingForClaims) return "§c现在不能碰";
        GamePlayer player = getGamePlayer(playerUuid);
        if (player == null) return "§c你不在该游戏中";
        if (players.indexOf(player) == lastDiscardPlayerIndex) return "§c不能碰自己打出的牌";
        if (!claimDecisions.containsKey(playerUuid)) return "§c你不能碰这张牌";
        if (player.countTile(lastDiscard) < 2) return "§c手中没有足够的牌来碰（需要2张 " + lastDiscard + "）";

        // 移除手中2张
        player.removeTile(lastDiscard);
        player.removeTile(lastDiscard);
        // 组成刻子副子（含弃牌共3张）
        Tile t = lastDiscard;
        Meld meld = new Meld(MeldType.PONG, Arrays.asList(new Tile(t.getSuit(), t.getNumber()),
                new Tile(t.getSuit(), t.getNumber()), new Tile(t.getSuit(), t.getNumber())));
        player.getMelds().add(meld);

        waitingForClaims = false;
        claimDecisions.clear();
        currentPlayerIndex = players.indexOf(player);

        broadcastMessage("§e" + player.getName() + " §f碰! " + meld.getDisplay());
        sendHandToPlayer(player);
        Player bp = Bukkit.getPlayer(playerUuid);
        if (bp != null) bp.sendMessage("§e请出牌 §7(/mj discard <牌序号或牌名>)");
        return null;
    }

    /**
     * 玩家杠牌：
     * - 等待认领时：明杠（从他人弃牌处杠）
     * - 轮到自己时：暗杠 或 补杠（从碰升级为杠）
     * @return null = 成功；否则为错误信息
     */
    public String gang(UUID playerUuid) {
        if (waitingForClaims) {
            return openKong(playerUuid);
        } else {
            return closedOrUpgradeKong(playerUuid);
        }
    }

    /** 明杠：从他人弃牌处杠 */
    private String openKong(UUID playerUuid) {
        GamePlayer player = getGamePlayer(playerUuid);
        if (player == null) return "§c你不在该游戏中";
        if (players.indexOf(player) == lastDiscardPlayerIndex) return "§c不能杠自己打出的牌";
        if (!claimDecisions.containsKey(playerUuid)) return "§c你不能杠这张牌";
        if (player.countTile(lastDiscard) < 3) return "§c手中没有足够的牌来杠（需要3张 " + lastDiscard + "）";

        for (int i = 0; i < 3; i++) player.removeTile(lastDiscard);
        Tile t = lastDiscard;
        List<Tile> kongTiles = Arrays.asList(new Tile(t.getSuit(), t.getNumber()),
                new Tile(t.getSuit(), t.getNumber()), new Tile(t.getSuit(), t.getNumber()),
                new Tile(t.getSuit(), t.getNumber()));
        Meld meld = new Meld(MeldType.KONG_OPEN, kongTiles);
        player.getMelds().add(meld);

        waitingForClaims = false;
        claimDecisions.clear();
        currentPlayerIndex = players.indexOf(player);

        broadcastMessage("§e" + player.getName() + " §f明杠! " + meld.getDisplay());
        drawAndPrompt(player);
        return null;
    }

    /** 暗杠 或 补杠（轮到自己时）*/
    private String closedOrUpgradeKong(UUID playerUuid) {
        GamePlayer player = getGamePlayer(playerUuid);
        if (player == null) return "§c你不在该游戏中";
        if (players.indexOf(player) != currentPlayerIndex) return "§c还没轮到你";

        // 优先检查暗杠（手中4张）
        for (TileSuit suit : TileSuit.values()) {
            for (int num = 1; num <= 9; num++) {
                Tile t = new Tile(suit, num);
                if (player.countTile(t) >= 4) {
                    for (int i = 0; i < 4; i++) player.removeTile(t);
                    List<Tile> kongTiles = Arrays.asList(new Tile(suit, num), new Tile(suit, num),
                            new Tile(suit, num), new Tile(suit, num));
                    player.getMelds().add(new Meld(MeldType.KONG_CLOSED, kongTiles));
                    broadcastMessage("§e" + player.getName() + " §f暗杠 " + t.getDisplay() + "!");
                    drawAndPrompt(player);
                    return null;
                }
            }
        }

        // 补杠（碰后摸到同牌）
        for (Meld m : new ArrayList<>(player.getMelds())) {
            if (m.getType() == MeldType.PONG) {
                Tile meldTile = m.getRepTile();
                if (player.hasTile(meldTile)) {
                    player.removeTile(meldTile);
                    player.getMelds().remove(m);
                    Tile t = meldTile;
                    List<Tile> kongTiles = Arrays.asList(new Tile(t.getSuit(), t.getNumber()),
                            new Tile(t.getSuit(), t.getNumber()), new Tile(t.getSuit(), t.getNumber()),
                            new Tile(t.getSuit(), t.getNumber()));
                    player.getMelds().add(new Meld(MeldType.KONG_OPEN, kongTiles));
                    broadcastMessage("§e" + player.getName() + " §f补杠 " + meldTile.getDisplay() + "!");
                    drawAndPrompt(player);
                    return null;
                }
            }
        }

        return "§c没有可以杠的牌（需要手中4张相同，或碰后摸到同牌）";
    }

    /**
     * 玩家宣告胡牌。
     * @return null = 成功；否则为错误信息
     */
    public String hu(UUID playerUuid) {
        if (state != GameState.PLAYING) return "§c游戏未在进行中";
        GamePlayer player = getGamePlayer(playerUuid);
        if (player == null) return "§c你不在该游戏中";

        if (waitingForClaims && claimDecisions.containsKey(playerUuid)) {
            // 放炮胡（他人弃牌）
            List<Tile> testHand = new ArrayList<>(player.getHand());
            testHand.add(lastDiscard);
            if (!WinChecker.canWin(testHand, player.getMelds())) {
                return "§c这不是有效的胡牌（需满足缺一门并能组成4组面子+1对将）";
            }
            player.getHand().add(lastDiscard);
            winner = player;
            lastWinSelfDraw = false;
            resolveWin(player, players.get(lastDiscardPlayerIndex), false);
            return null;

        } else if (!waitingForClaims && players.indexOf(player) == currentPlayerIndex) {
            // 自摸胡
            if (!WinChecker.canWin(player.getHand(), player.getMelds())) {
                return "§c这不是有效的胡牌（需满足缺一门并能组成4组面子+1对将）";
            }
            winner = player;
            lastWinSelfDraw = true;
            resolveWin(player, null, true);
            return null;
        }

        return "§c现在不能胡牌";
    }

    /**
     * 玩家放弃本次认领机会（pass）。
     * @return null = 成功；否则为错误信息
     */
    public String pass(UUID playerUuid) {
        if (!waitingForClaims) return "§c现在没有可放弃的操作";
        if (!claimDecisions.containsKey(playerUuid)) return "§c你不需要决定";
        claimDecisions.put(playerUuid, true); // 已决定（放弃）

        Player bp = Bukkit.getPlayer(playerUuid);
        if (bp != null) bp.sendMessage("§7已放弃本次操作");

        // 如果所有人都已决定
        boolean allDecided = claimDecisions.values().stream().allMatch(v -> v);
        if (allDecided) {
            waitingForClaims = false;
            claimDecisions.clear();
            advanceToNextPlayer();
        }
        return null;
    }

    // ---- Win resolution ----

    private void resolveWin(GamePlayer winnerPlayer, GamePlayer discardPlayer, boolean selfDraw) {
        waitingForClaims = false;
        state = GameState.ENDED;

        int fan = WinChecker.calculateFan(winnerPlayer.getHand(), winnerPlayer.getMelds(), selfDraw);
        int basePoints = 8 * fan;

        broadcastMessage("§6§l======== " + winnerPlayer.getName() + " 胡牌！========");
        // 特殊牌型提示
        if (WinChecker.isQingYiSe(winnerPlayer.getHand(), winnerPlayer.getMelds()))
            broadcastMessage("§6§l★ 清一色！");
        else if (WinChecker.isPengPengHu(winnerPlayer.getHand(), winnerPlayer.getMelds()))
            broadcastMessage("§6§l★ 碰碰胡！");
        if (selfDraw)
            broadcastMessage("§6§l★ 自摸！");
        broadcastMessage("§e番数: §f" + fan + "番  §e基础分: §f" + basePoints + "分");

        // 展示胡牌手牌
        broadcastMessage("§e副子: " + winnerPlayer.getMeldsDisplay());
        broadcastMessage("§e手牌: " + winnerPlayer.getHandDisplay());

        // 计分
        if (selfDraw) {
            for (GamePlayer p : players) {
                if (!p.getUuid().equals(winnerPlayer.getUuid())) {
                    p.addScore(-basePoints);
                    winnerPlayer.addScore(basePoints);
                }
            }
        } else {
            // 放炮者赔双倍
            discardPlayer.addScore(-basePoints * 2);
            winnerPlayer.addScore(basePoints * 2);
            broadcastMessage("§c" + discardPlayer.getName() + " 放炮，赔 " + (basePoints * 2) + "分");
        }

        // 显示积分
        broadcastMessage("§e======= 当前积分 =======");
        for (GamePlayer p : players) {
            broadcastMessage("  §f" + p.getName() + " (" + p.getSeatWindName() + "家): §e" + p.getScore() + "分");
        }
        broadcastMessage("§7输入 §f/mj create §7或 §f/mj join §7开始新游戏");
    }

    // ---- Turn management ----

    /** 轮到下一位玩家摸牌 */
    private void advanceToNextPlayer() {
        currentPlayerIndex = (currentPlayerIndex + 1) % players.size();
        GamePlayer next = getCurrentPlayer();

        if (wall.isEmpty()) {
            broadcastMessage("§c牌墙已空，本局流局！");
            broadcastMessage("§e======= 当前积分 =======");
            for (GamePlayer p : players) {
                broadcastMessage("  §f" + p.getName() + " (" + p.getSeatWindName() + "家): §e" + p.getScore() + "分");
            }
            state = GameState.ENDED;
            return;
        }

        Tile drawn = wall.poll();
        next.getHand().add(drawn);
        broadcastMessage("§7" + next.getName() + " 摸牌  §7(牌墙剩余: " + wall.size() + "张)");

        Player bp = Bukkit.getPlayer(next.getUuid());
        if (bp != null) {
            bp.sendMessage("§e摸牌: " + drawn.getDisplay());
            sendHandToPlayer(next);
            // 检查自摸
            if (WinChecker.canWin(next.getHand(), next.getMelds())) {
                bp.sendMessage("§6§l你可以自摸胡牌！ §f(/mj hu)");
            }
            // 提示可能的杠
            hintSelfKong(next, bp);
            bp.sendMessage("§e请出牌 §7(/mj discard <牌序号或牌名>)");
        }
    }

    /** 杠后补摸一张牌，并提示操作 */
    private void drawAndPrompt(GamePlayer player) {
        if (wall.isEmpty()) {
            broadcastMessage("§c牌墙已空，本局流局！");
            state = GameState.ENDED;
            return;
        }
        Tile drawn = wall.poll();
        player.getHand().add(drawn);
        Player bp = Bukkit.getPlayer(player.getUuid());
        if (bp != null) {
            bp.sendMessage("§e杠后补摸: " + drawn.getDisplay());
            sendHandToPlayer(player);
            if (WinChecker.canWin(player.getHand(), player.getMelds())) {
                bp.sendMessage("§6§l你可以自摸胡牌！ §f(/mj hu)");
            }
            hintSelfKong(player, bp);
            bp.sendMessage("§e请出牌 §7(/mj discard <牌序号或牌名>)");
        }
    }

    /** 提示当前玩家可以进行的暗杠/补杠 */
    private void hintSelfKong(GamePlayer player, Player bp) {
        // 暗杠
        for (TileSuit suit : TileSuit.values()) {
            for (int num = 1; num <= 9; num++) {
                Tile t = new Tile(suit, num);
                if (player.countTile(t) >= 4) {
                    bp.sendMessage("§a可以暗杠 " + t.getDisplay() + " §7(/mj gang)");
                }
            }
        }
        // 补杠
        for (Meld m : player.getMelds()) {
            if (m.getType() == MeldType.PONG && player.hasTile(m.getRepTile())) {
                bp.sendMessage("§a可以补杠 " + m.getRepTile().getDisplay() + " §7(/mj gang)");
            }
        }
    }

    // ---- Utility ----

    /** 向玩家发送当前手牌信息 */
    public void sendHandToPlayer(GamePlayer player) {
        Player bp = Bukkit.getPlayer(player.getUuid());
        if (bp == null) return;
        bp.sendMessage("§6======= 你的手牌 =======");
        bp.sendMessage("§e副子: " + player.getMeldsDisplay());
        bp.sendMessage("§e手牌(" + player.getHand().size() + "张): " + player.getHandDisplay());
        bp.sendMessage("§7(出牌: /mj discard <序号> 或 <牌名如1wan/9tiao/5bing>)");
    }

    /** 向房间内所有在线玩家广播消息 */
    public void broadcastMessage(String message) {
        for (GamePlayer p : players) {
            Player bp = Bukkit.getPlayer(p.getUuid());
            if (bp != null) bp.sendMessage(message);
        }
    }
}
