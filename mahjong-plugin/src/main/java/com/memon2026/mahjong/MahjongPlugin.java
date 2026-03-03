package com.memon2026.mahjong;

import com.memon2026.mahjong.command.MahjongCommand;
import com.memon2026.mahjong.manager.GameManager;
import org.bukkit.plugin.java.JavaPlugin;

/**
 * 广东麻将 Minecraft Spigot 插件主类
 *
 * 功能：
 * - 108 张牌：万/条/饼各 1-9 点，每种 4 副
 * - 最多 4 人游戏
 * - 广东麻将规则：缺一门、不允许吃牌、碰/杠/胡
 * - 支持暗杠、明杠、补杠
 * - 自摸/放炮计分
 * - 特殊牌型：清一色、碰碰胡
 */
public class MahjongPlugin extends JavaPlugin {

    private GameManager gameManager;

    @Override
    public void onEnable() {
        gameManager = new GameManager();

        // 注册命令
        MahjongCommand mahjongCommand = new MahjongCommand(gameManager);
        getCommand("mahjong").setExecutor(mahjongCommand);
        getCommand("mahjong").setTabCompleter(mahjongCommand);

        getLogger().info("========================================");
        getLogger().info("  广东麻将插件已启动！");
        getLogger().info("  使用 /mahjong 或 /mj 开始游戏");
        getLogger().info("========================================");
    }

    @Override
    public void onDisable() {
        getLogger().info("广东麻将插件已卸载");
    }

    public GameManager getGameManager() {
        return gameManager;
    }
}
