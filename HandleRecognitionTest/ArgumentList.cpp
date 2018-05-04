#include <QApplication>
#include "ArgumentList.h"

ArgumentList::ArgumentList()
{
    *this = qApp->arguments();
}

void ArgumentList::argsToStringlist(int argc, char* argv[])
{
    for (int i = 0; i < argc; ++i) {
        *this += argv[i];
    }
}

bool ArgumentList::getSwitch(QString option) 
{
    QMutableStringListIterator itr(*this);

    while (itr.hasNext()) {
        if (option == itr.next()) {
            itr.remove();
            return true;
        }
    }

    return false;
}

QString ArgumentList::getSwitchArg(QString option, QString defaultRetValue) 
{
    if (isEmpty())
        return defaultRetValue;

    QMutableStringListIterator itr(*this);

    while (itr.hasNext()) {
        if (option == itr.next()) {
            itr.remove();
            QString retval = itr.next();
            itr.remove();
            return retval;
        }
    }

    return defaultRetValue;
}
