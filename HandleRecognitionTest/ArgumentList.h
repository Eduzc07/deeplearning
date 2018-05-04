/* <Simple class to deal with the command line arguments> */
#ifndef ARGUMENTLIST_H
#define ARGUMENTLIST_H

#include <QStringList>

///Taken from Paul Ezust's Design Patterns with QT book
class ArgumentList : public QStringList
{
public:
    ArgumentList();
    ArgumentList(int argc, char* argv[]) { argsToStringlist(argc, argv); }
    ArgumentList(const QStringList& argumentList):QStringList(argumentList) {}

    bool getSwitch(QString option);
    QString getSwitchArg(QString option, QString defaultRetValue = QString());

private:
    void argsToStringlist(int argc, char* argv[]);
};

#endif /// ARGUMENTLIST_H
