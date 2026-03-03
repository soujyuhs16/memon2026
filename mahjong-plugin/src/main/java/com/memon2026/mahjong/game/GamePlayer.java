package com.memon2026.mahjong.game;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;

/**
 * 游戏中的玩家，持有手牌、副子和积分信息。
 */
public class GamePlayer {

    private static final String[] WIND_NAMES = {"东", "南", "西", "北"};

    private final UUID uuid;
    private final String name;
    private final List<Tile> hand;
    private final List<Meld> melds;
    private int score;
    private int seatWind; // 0=东 1=南 2=西 3=北

    public GamePlayer(UUID uuid, String name) {
        this.uuid = uuid;
        this.name = name;
        this.hand = new ArrayList<>();
        this.melds = new ArrayList<>();
        this.score = 1000;
    }

    // ---- Getters ----

    public UUID getUuid() { return uuid; }
    public String getName() { return name; }
    public List<Tile> getHand() { return hand; }
    public List<Meld> getMelds() { return melds; }
    public int getScore() { return score; }
    public int getSeatWind() { return seatWind; }
    public String getSeatWindName() { return WIND_NAMES[seatWind]; }

    // ---- Setters ----

    public void setSeatWind(int seatWind) { this.seatWind = seatWind; }
    public void setScore(int score) { this.score = score; }
    public void addScore(int delta) { this.score += delta; }

    // ---- Hand operations ----

    /** 对手牌排序 */
    public void sortHand() {
        Collections.sort(hand);
    }

    /**
     * 从手牌中移除一张指定的牌（移除第一个匹配项）。
     * @return 是否成功移除
     */
    public boolean removeTile(Tile tile) {
        for (int i = 0; i < hand.size(); i++) {
            if (hand.get(i).equals(tile)) {
                hand.remove(i);
                return true;
            }
        }
        return false;
    }

    /** 统计手牌中某张牌的数量 */
    public int countTile(Tile tile) {
        int count = 0;
        for (Tile t : hand) if (t.equals(tile)) count++;
        return count;
    }

    public boolean hasTile(Tile tile) { return countTile(tile) >= 1; }
    public boolean hasThreeOfTile(Tile tile) { return countTile(tile) >= 2; } // 手中2张 + 他人弃牌1张
    public boolean hasFourOfTile(Tile tile) { return countTile(tile) >= 3; }  // 手中3张 + 他人弃牌1张

    // ---- Display ----

    /**
     * 返回带编号的手牌显示字符串，方便玩家通过序号选择出牌。
     * 格式：§71:§e1万§r §72:§e2万§r ...
     */
    public String getHandDisplay() {
        sortHand();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < hand.size(); i++) {
            sb.append("§7").append(i + 1).append(":").append(hand.get(i).getDisplay()).append(" ");
        }
        return sb.toString().trim();
    }

    /** 返回副子的显示字符串 */
    public String getMeldsDisplay() {
        if (melds.isEmpty()) return "§7（无）";
        StringBuilder sb = new StringBuilder();
        for (Meld m : melds) sb.append(m.getDisplay()).append(" ");
        return sb.toString().trim();
    }

    /** 通过 1-based 索引获取手牌 */
    public Tile getTileByIndex(int oneBasedIndex) {
        sortHand();
        if (oneBasedIndex < 1 || oneBasedIndex > hand.size()) return null;
        return hand.get(oneBasedIndex - 1);
    }
}
