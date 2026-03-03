package com.memon2026.mahjong.game;

import java.util.Objects;

/**
 * 麻将牌 (Mahjong Tile)
 * 广东麻将使用 108 张牌：万/条/饼各 1-9 点，每种 4 张。
 */
public class Tile implements Comparable<Tile> {

    private final TileSuit suit;
    private final int number; // 1-9

    public Tile(TileSuit suit, int number) {
        if (number < 1 || number > 9) {
            throw new IllegalArgumentException("Tile number must be 1-9, got: " + number);
        }
        this.suit = suit;
        this.number = number;
    }

    public TileSuit getSuit() { return suit; }
    public int getNumber() { return number; }

    /** 带颜色的聊天显示格式 */
    public String getDisplay() {
        return suit.getColor() + number + suit.getDisplay() + "§r";
    }

    /** 简短字符串表示，如 "1万" "9条" */
    @Override
    public String toString() {
        return number + suit.getDisplay();
    }

    /** 用于排序的索引值 (0-26) */
    public int getIndex() {
        return suit.ordinal() * 9 + (number - 1);
    }

    @Override
    public int compareTo(Tile other) {
        return Integer.compare(this.getIndex(), other.getIndex());
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Tile)) return false;
        Tile tile = (Tile) o;
        return number == tile.number && suit == tile.suit;
    }

    @Override
    public int hashCode() {
        return Objects.hash(suit, number);
    }

    /**
     * 从字符串解析牌，支持格式：
     * - "1wan" / "1万" (万子)
     * - "9tiao" / "9条" (条子)
     * - "5bing" / "5饼" (饼子)
     * - 也支持简写 "1w", "9t", "5b"
     */
    public static Tile fromString(String s) {
        if (s == null || s.isEmpty()) {
            throw new IllegalArgumentException("Empty tile string");
        }
        s = s.trim().toLowerCase();
        int num;
        TileSuit suit;
        try {
            if (s.endsWith("wan") || s.endsWith("万")) {
                int suffixLen = s.endsWith("wan") ? 3 : 1;
                num = Integer.parseInt(s.substring(0, s.length() - suffixLen));
                suit = TileSuit.WAN;
            } else if (s.endsWith("tiao") || s.endsWith("条")) {
                int suffixLen = s.endsWith("tiao") ? 4 : 1;
                num = Integer.parseInt(s.substring(0, s.length() - suffixLen));
                suit = TileSuit.TIAO;
            } else if (s.endsWith("bing") || s.endsWith("饼")) {
                int suffixLen = s.endsWith("bing") ? 4 : 1;
                num = Integer.parseInt(s.substring(0, s.length() - suffixLen));
                suit = TileSuit.BING;
            } else if (s.endsWith("w")) {
                num = Integer.parseInt(s.substring(0, s.length() - 1));
                suit = TileSuit.WAN;
            } else if (s.endsWith("t")) {
                num = Integer.parseInt(s.substring(0, s.length() - 1));
                suit = TileSuit.TIAO;
            } else if (s.endsWith("b")) {
                num = Integer.parseInt(s.substring(0, s.length() - 1));
                suit = TileSuit.BING;
            } else {
                throw new IllegalArgumentException("Unknown tile format: " + s
                        + ". Use formats like: 1wan/1w/1万, 9tiao/9t/9条, 5bing/5b/5饼");
            }
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Invalid tile number in: " + s);
        }
        return new Tile(suit, num);
    }
}
