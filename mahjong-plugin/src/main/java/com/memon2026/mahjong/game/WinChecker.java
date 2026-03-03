package com.memon2026.mahjong.game;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * 广东麻将胡牌判断器
 *
 * 广东麻将胡牌规则：
 * 1. 缺一门：手牌（含副子）必须缺少至少一种花色
 * 2. 标准胡牌：4组面子（顺子或刻子）+ 1对将牌
 *    - 注意：手牌中可以有顺子（自摸组成），但不能"吃"他人弃牌
 * 3. 特殊牌型：
 *    - 碰碰胡（全刻）：所有面子均为刻子
 *    - 清一色：所有牌同一花色
 */
public class WinChecker {

    /**
     * 判断当前手牌是否能胡牌。
     * 调用前 hand 的大小应满足：hand.size() == (4 - melds.size()) * 3 + 2
     *
     * @param hand  手牌列表（已包含刚摸/获得的那张牌，共 14 - meld tiles 张）
     * @param melds 已公开的副子列表
     * @return 是否胡牌
     */
    public static boolean canWin(List<Tile> hand, List<Meld> melds) {
        int setsFromMelds = melds.size();
        int remainingSets = 4 - setsFromMelds;
        int expectedHandSize = remainingSets * 3 + 2;
        if (hand.size() != expectedHandSize) return false;

        // 缺一门判断
        if (!queYiMen(hand, melds)) return false;

        // 转为计数数组，尝试组牌
        int[] counts = toCounts(hand);
        return canFormWinningHand(counts, remainingSets);
    }

    /**
     * 缺一门检查：手牌+副子中存在的花色种数必须少于 3。
     */
    public static boolean queYiMen(List<Tile> hand, List<Meld> melds) {
        Set<TileSuit> suits = new HashSet<>();
        for (Tile t : hand) suits.add(t.getSuit());
        for (Meld m : melds) {
            for (Tile t : m.getTiles()) suits.add(t.getSuit());
        }
        return suits.size() < 3;
    }

    /**
     * 碰碰胡（全刻）检查：所有面子（含副子）均为刻子，无顺子。
     */
    public static boolean isPengPengHu(List<Tile> hand, List<Meld> melds) {
        // 副子必须全为碰或杠
        for (Meld m : melds) {
            if (m.getType() == MeldType.KONG_OPEN || m.getType() == MeldType.KONG_CLOSED
                    || m.getType() == MeldType.PONG) {
                continue;
            }
            return false;
        }
        // 手牌部分尝试只用刻子组成
        int setsNeeded = 4 - melds.size();
        int[] counts = toCounts(hand);
        return canFormAllTriplets(counts, setsNeeded);
    }

    /**
     * 清一色检查：所有牌（含副子）同一花色。
     */
    public static boolean isQingYiSe(List<Tile> hand, List<Meld> melds) {
        Set<TileSuit> suits = new HashSet<>();
        for (Tile t : hand) suits.add(t.getSuit());
        for (Meld m : melds) {
            for (Tile t : m.getTiles()) suits.add(t.getSuit());
        }
        return suits.size() == 1;
    }

    /**
     * 计算番数（用于积分）。
     * 基础：1番；自摸：+1番；清一色：+3番；碰碰胡：+1番（与清一色互斥取高）。
     */
    public static int calculateFan(List<Tile> hand, List<Meld> melds, boolean isSelfDraw) {
        int fan = 1;
        if (isSelfDraw) fan += 1;
        if (isQingYiSe(hand, melds)) {
            fan += 3;
        } else if (isPengPengHu(hand, melds)) {
            fan += 1;
        }
        return fan;
    }

    // -----------------------------------------------------------------------
    // 内部工具方法
    // -----------------------------------------------------------------------

    /**
     * 将牌列表转换为计数数组（长度 27）。
     * 索引 = suit.ordinal() * 9 + (number - 1)，对应万1…万9，条1…条9，饼1…饼9。
     */
    public static int[] toCounts(List<Tile> tiles) {
        int[] counts = new int[27];
        for (Tile t : tiles) {
            counts[t.getSuit().ordinal() * 9 + (t.getNumber() - 1)]++;
        }
        return counts;
    }

    /**
     * 尝试从计数数组中组成"setsNeeded 组面子 + 1 对将牌"。
     * 遍历所有可能的将牌，剩余递归检查是否能凑满面子。
     */
    private static boolean canFormWinningHand(int[] counts, int setsNeeded) {
        for (int i = 0; i < 27; i++) {
            if (counts[i] >= 2) {
                counts[i] -= 2;
                if (canFormSetsOnly(counts, setsNeeded)) {
                    counts[i] += 2;
                    return true;
                }
                counts[i] += 2;
            }
        }
        return false;
    }

    /**
     * 递归检查计数数组能否凑成恰好 setsNeeded 组面子（顺子或刻子）。
     */
    private static boolean canFormSetsOnly(int[] counts, int setsNeeded) {
        if (setsNeeded == 0) {
            for (int c : counts) if (c != 0) return false;
            return true;
        }
        // 找第一张非零牌
        int first = -1;
        for (int i = 0; i < 27; i++) {
            if (counts[i] > 0) { first = i; break; }
        }
        if (first == -1) return false;

        // 尝试刻子（3张相同）
        if (counts[first] >= 3) {
            counts[first] -= 3;
            if (canFormSetsOnly(counts, setsNeeded - 1)) {
                counts[first] += 3;
                return true;
            }
            counts[first] += 3;
        }

        // 尝试顺子（3张连续同花色）
        int suit = first / 9;
        int num = first % 9; // 0 = 1点, 8 = 9点
        if (num <= 6                               // 至少还有两张可连续
                && (first + 1) / 9 == suit         // 第2张同花色
                && (first + 2) / 9 == suit         // 第3张同花色
                && counts[first + 1] > 0
                && counts[first + 2] > 0) {
            counts[first]--;
            counts[first + 1]--;
            counts[first + 2]--;
            if (canFormSetsOnly(counts, setsNeeded - 1)) {
                counts[first]++;
                counts[first + 1]++;
                counts[first + 2]++;
                return true;
            }
            counts[first]++;
            counts[first + 1]++;
            counts[first + 2]++;
        }

        return false;
    }

    /**
     * 检查计数数组能否凑成 setsNeeded 组刻子 + 1 对将牌（碰碰胡用）。
     */
    private static boolean canFormAllTriplets(int[] counts, int setsNeeded) {
        for (int i = 0; i < 27; i++) {
            if (counts[i] >= 2) {
                counts[i] -= 2;
                if (canFormTripletsOnly(counts, setsNeeded)) {
                    counts[i] += 2;
                    return true;
                }
                counts[i] += 2;
            }
        }
        return false;
    }

    private static boolean canFormTripletsOnly(int[] counts, int setsNeeded) {
        if (setsNeeded == 0) {
            for (int c : counts) if (c != 0) return false;
            return true;
        }
        int first = -1;
        for (int i = 0; i < 27; i++) {
            if (counts[i] > 0) { first = i; break; }
        }
        if (first == -1) return false;
        if (counts[first] < 3) return false;
        counts[first] -= 3;
        boolean result = canFormTripletsOnly(counts, setsNeeded - 1);
        counts[first] += 3;
        return result;
    }
}
