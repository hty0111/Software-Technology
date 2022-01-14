DROP TABLE IF EXISTS `stu`;
CREATE TABLE `stu` (
    姓名 varchar(8),
    性别 varchar(4),
    电子邮箱 varchar(50)
) ENGINE=InnoDB CHARSET=utf8;
ALTER TABLE `stu` ADD 专业 varchar(16) CHARACTER SET utf8;
ALTER TABLE `stu` DROP COLUMN 性别;
ALTER TABLE `stu` ADD 学号 varchar(32) CHARACTER SET utf8;
INSERT INTO `stu` (学号,姓名,电子邮箱,专业) VALUES (31700001,'唐三藏','datangsanzang@zju.edu.cn','工信'),(31600002,'猪八戒','tianpenglaozhu@zju.edu.cn','工信'),(31500004,'白龙马','xihaisantaizi@zju.edu.cn','金融');
UPDATE `stu` SET 专业='工信' WHERE 学号 like '315%';
DELETE FROM `stu` WHERE (姓名 like '猪%' AND 专业='工信');
SELECT 姓名 FROM `stu` WHERE 专业='工信';
